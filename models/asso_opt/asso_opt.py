import os
import torchmetrics
import wandb
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class AssoConcept(pl.LightningModule):
    def init_weight_concept(self, concept2cls):
        self.init_weight = th.zeros((self.cfg.num_cls, len(self.select_idx))) #init with the actual number of selected index

        if self.cfg.use_rand_init: th.nn.init.kaiming_normal_(self.init_weight)
        else: self.init_weight.scatter_(0, concept2cls, self.cfg.init_val)
            
        if 'cls_name_init' in self.cfg and self.cfg.cls_name_init != 'none':
            if self.cfg.cls_name_init == 'replace':
                self.init_weight = th.load(self.init_weight_save_dir)
            elif self.cfg.cls_name_init == 'combine':
                self.init_weight += th.load(self.init_weight_save_dir)
                self.init_weight = self.init_weight.clip(max=1)
            elif self.cfg.cls_name_init == 'random':
                th.nn.init.kaiming_normal_(self.init_weight)


    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')
        self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        cls_sim_path = self.data_root.joinpath('cls_sim.pth')

        if not concept_feat_path.exists():
            raise RuntimeError('need to call datamodule precompute_txt before using the model')
        else:
            if select_idx is None: self.select_idx = th.load(select_idx_path)[:cfg.num_concept]
            else: self.select_idx = select_idx

            self.concepts = th.load(concept_feat_path)[self.select_idx].cuda()
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            self.concept_raw = np.load(concept_raw_path)[self.select_idx]
            self.concept2cls = th.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1)

        if init_weight is None:
            self.init_weight_concept(self.concept2cls)
        else:
            self.init_weight = init_weight

        if 'cls_sim_prior' in self.cfg and self.cfg.cls_sim_prior and self.cfg.cls_sim_prior != 'none':
            # class similarity is prior to restrict class-concept association
            # if class A and B are dissimilar (similarity=0), then the mask location will be 0 
            print('use cls prior')
            print(cls_sim_path)
            cls_sim = th.load(cls_sim_path)
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                print(concept_id)
                print(self.init_weight.shape)
                print(self.init_weight[:,concept_id])
                target_class = int(th.where(self.init_weight[:,concept_id] == 1)[0])
                new_weights.append(cls_sim[target_class] + self.init_weight[:,concept_id])
            
            os._exit(1)
            self.init_weight = th.vstack(new_weights).T
            # self.weight_mask = cls_sim @ self.init_weight

        self.asso_mat = th.nn.Parameter(self.init_weight.clone())
        
        print(f"Num labels is: {cfg.num_cls}")
        
        if 'XRAY' in self.cfg.data_root:
            # we're doing multi-label classification, use F1 instead of accuracy
            # self.train_acc = torchmetrics.Accuracy(num_labels=cfg.num_cls, task='multilabel')
            # self.valid_acc = torchmetrics.Accuracy(num_labels=cfg.num_cls, task='multilabel')
            # self.test_acc = torchmetrics.Accuracy(num_labels=cfg.num_cls, task='multilabel')
            self.train_f1 = torchmetrics.F1Score(num_labels=cfg.num_cls, task='multilabel', average='macro')
            self.valid_f1 = torchmetrics.F1Score(num_labels=cfg.num_cls, task='multilabel', average='macro')
            self.test_f1 = torchmetrics.F1Score(num_labels=cfg.num_cls, task='multilabel', average='macro')
        else:        
            self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
            self.valid_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
            # self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, average='macro')
            self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
        self.all_y = []
        self.all_pred = []
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.cfg.num_cls, task='multiclass')
        
        if 'XRAY' in self.cfg.data_root:
            self.logit_scale = th.nn.Parameter(th.ones([]) * np.log(1 / 0.07))
        
        self.clip_model = self.cfg.clip_model
        self.save_hyperparameters()


    def _get_weight_mat(self):
        if self.cfg.asso_act == 'relu':
            mat = F.relu(self.asso_mat)
        elif self.cfg.asso_act == 'tanh':
            mat = F.tanh(self.asso_mat) 
        elif self.cfg.asso_act == 'softmax':
            mat = F.softmax(self.asso_mat, dim=-1) 
        else:
            mat = self.asso_mat
        return mat 


    def forward(self, img_feat):
        mat = self._get_weight_mat()
        cls_feat = mat @ self.concepts
        sim = img_feat @ cls_feat.t()
        th.set_printoptions(threshold=10000)
        print(f"Sim: {sim}")
        return sim


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        print(f"Data Root: {self.cfg.data_root}")

        sim = self.forward(image)
        if 'XRAY' in self.cfg.data_root:
            pred = sim  # do not multiply by 100
        else:
            pred = 100 * sim  # scaling as standard CLIP does
                
        # pred = sim

        # classification accuracy
        # TODO: update this for multi-label classification
        # print(pred.shape)
        # print(label.shape)
        
        # print(f"Pred: {pred}")
        # print(f"Label: {label}")

        if 'XRAY' in self.cfg.data_root:
            # we need to do multi-label classification
            # print('doing binary cross entropy with logits')
            cls_loss = F.binary_cross_entropy_with_logits(pred, label.float())
            # probs = th.sigmoid(pred)
        else:
            cls_loss = F.cross_entropy(pred, label)
            
        print(f"cls_loss: {cls_loss}")
        
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        self.log('training_loss', cls_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)

        # run sigmoid then threshold pred
        if 'XRAY' in self.cfg.data_root:
            print('doing sigmoid then threshold')
            pred_label = (th.sigmoid(pred) > 0.5).float()
            print(pred)
            # os._exit(1)
            print(f"train f1: {self.train_f1(pred_label, label)}")
            self.train_f1(pred_label, label)
        else:
            pred_label = pred
            self.train_acc(pred_label, label)
            print(f"train accuracy: {self.train_acc(pred_label, label)}")
            self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss


    def configure_optimizers(self):
        opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt


    def validation_step(self, batch, batch_idx):        
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                # wandb.define_metric('val_acc', summary='max')
                if 'XRAY' in self.cfg.data_root:
                    wandb.define_metric('val_f1', summary='max')
                else:
                    wandb.define_metric('val_acc', summary='max')
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        if 'XRAY' in self.cfg.data_root:
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
            probs = th.sigmoid(pred)
            pred_label = (probs > 0.5).float()
            self.valid_f1(pred_label, y)
            self.log('val_f1', self.valid_f1, on_step=False, on_epoch=True)
            self.log('val_loss', loss)
        else:
            loss = F.cross_entropy(pred, y)
            self.log('val_loss', loss)
            self.valid_acc(pred, y)
            self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        print('running test step')
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        if 'XRAY' in self.cfg.data_root:
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
            probs = th.sigmoid(pred)
            pred_label = (probs > 0.5).float()
            self.test_f1(pred_label, y)
            print(f"test f1: {self.test_f1(pred_label, y)}")
            self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
            self.log('test_loss', loss)
        else:
            loss = F.cross_entropy(pred, y)
            y_pred = pred.argmax(dim=-1)
            self.confmat(y_pred, y)
            self.all_y.append(y)
            self.all_pred.append(y_pred)
            self.log('test_loss', loss)
            self.test_acc(pred, y)
            print(f"test accuracy: {self.test_acc(pred, y)}")
            self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss


    # def test_epoch_end(self, outputs):
    #     all_y = th.hstack(self.all_y)
    #     all_pred = th.hstack(self.all_pred)
    #     self.total_test_acc = self.test_acc(all_pred, all_y)
    #     pass

    def on_test_epoch_end(self):
        all_y = th.hstack(self.all_y)
        all_pred = th.hstack(self.all_pred)
        if 'XRAY' in self.cfg.data_root:
            self.total_test_f1 = self.test_f1(all_pred, all_y)
        else:
            self.total_test_acc = self.test_acc(all_pred, all_y)
        pass

    def on_predict_epoch_start(self):
        self.num_pred = 4
        self.concepts = self.concepts.to(self.device)

        self.pred_table = wandb.Table(
            columns=["img", "label"] +
            ["pred_{}".format(i) for i in range(self.num_pred)])


    def predict_step(self, batch, batch_idx):
        image, y, image_name = batch
        sim = self.forward(image)
        pred = 100 * sim
        _, y_pred = th.topk(pred, self.num_pred)
        for img_path, gt, top_pred in zip(image_name, y, y_pred):
            gt = (gt + 1).item()
            top_pred = (top_pred + 1).tolist()
            self.pred_table.add_data(wandb.Image(img_path), gt, *top_pred)
    

    def on_predict_epoch_end(self, results):
        wandb.log({'pred_table': self.pred_table})
    

    def prune_asso_mat(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat().detach()
        val_asso_mat = th.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = th.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good


    def extract_cls2concept(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat().detach()
        strong_asso = asso_mat > thresh 
        res = {}
        import pdb; pdb.set_trace()
        for i, cls_name in enumerate(cls_names): 
            ## threshold globally
            keep_idx = strong_asso[i]
            ## sort
            res[cls_name] = np.unique(self.concept_raw[keep_idx])
        return res


    def extract_concept2cls(self, percent_thresh=0.95, mode='global'):
        asso_mat = self.asso_mat.detach()
        res = {} 
        for i in range(asso_mat.shape[1]):
            res[i] = th.argsort(asso_mat[:, i], descending=True).tolist()
        return res


class AssoConceptFast(AssoConcept):

    def forward(self, dot_product):
        print('running fast')
        mat = self._get_weight_mat()
        # if "XRAY" in self.cfg.data_root:
        #     mat = F.normalize(mat, p=2, dim=-1)  # normalize each row vector
        if "XRAY" in self.cfg.data_root:
            res = dot_product @ mat.t() * self.logit_scale.exp()
        else:
            res = dot_product @ mat.t()
        th.set_printoptions(threshold=10000)
        print(f"Res: {res}")
        return res
    
# def AssoConceptMoE