import os
import torchmetrics
import wandb
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class AssoConceptMoE(pl.LightningModule):
    """
    Mixture-of-Experts (MoE) type implementation
    """

    def init_weight_concept_specialist(self, concept2cls):
        self.init_weight_specialist = th.zeros(
            (self.cfg.num_cls, len(self.select_idx_specialist))
        )  # init with the actual number of selected index

        if self.cfg.use_rand_init:
            th.nn.init.kaiming_normal_(self.init_weight_specialist)
        else:
            self.init_weight_specialist.scatter_(0, concept2cls, self.cfg.init_val)

        if "cls_name_init" in self.cfg and self.cfg.cls_name_init != "none":
            if self.cfg.cls_name_init == "replace":
                self.init_weight_specialist = th.load(
                    self.init_weight_save_dir_specialist
                )
            elif self.cfg.cls_name_init == "combine":
                self.init_weight_specialist += th.load(
                    self.init_weight_save_dir_specialist
                )
                self.init_weight_specialist = self.init_weight_specialist.clip(max=1)
            elif self.cfg.cls_name_init == "random":
                th.nn.init.kaiming_normal_(self.init_weight_specialist)

    def init_weight_concept_generalist(self, concept2cls):
        self.init_weight_generalist = th.zeros(
            (self.cfg.num_cls, len(self.select_idx_generalist))
        )

        if self.cfg.use_rand_init:
            th.nn.init.kaiming_normal_(self.init_weight_generalist)
        else:
            self.init_weight_generalist.scatter_(0, concept2cls, self.cfg.init_val)

        if "cls_name_init" in self.cfg and self.cfg.cls_name_init != "none":
            if self.cfg.cls_name_init == "replace":
                self.init_weight_specialist = th.load(
                    self.init_weight_save_dir_generalist
                )
            elif self.cfg.cls_name_init == "combine":
                self.init_weight_generalist += th.load(
                    self.init_weight_save_dir_generalist
                )
                self.init_weight_generalist = self.init_weight_generalist.clip(max=1)
            elif self.cfg.cls_name_init == "random":
                th.nn.init.kaiming_normal_(self.init_weight_generalist)

    def __init__(
        self,
        cfg,
        init_weight_generalist=None,
        init_weight_specialist=None,
        select_idx_generalist=None,
        select_idx_specialist=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)

        # Grab both generalist and specialist concepts
        concept_feat_path_generalist = self.data_root.joinpath(
            "concepts_feat_{}.pth".format(self.cfg.clip_model.replace("/", "-"))
        )
        concept_raw_path_generalist = self.data_root.joinpath(
            "concepts_raw_selected.npy"
        )
        concept2cls_path_generalist = self.data_root.joinpath(
            "concept2cls_selected.npy"
        )
        select_idx_path_generalist = self.data_root.joinpath("select_idx.pth")

        concept_feat_path_specialist = self.data_root.joinpath(
            "concepts_feat_specialist_{}.pth".format(
                self.cfg.clip_model.replace("/", "-")
            )
        )
        concept_raw_path_specialist = self.data_root.joinpath(
            "concepts_raw_selected_specialist.npy"
        )
        concept2cls_path_specialist = self.data_root.joinpath(
            "concept2cls_selected_specialist.npy"
        )
        select_idx_path_specialist = self.data_root.joinpath(
            "select_idx_specialist.pth"
        )

        self.init_weight_save_dir_generalist = self.data_root.joinpath(
            "init_weight_generalist.pth"
        )
        cls_sim_path_generalist = self.data_root.joinpath("cls_sim_generalist.pth")

        self.init_weight_save_dir_specialist = self.data_root.joinpath(
            "init_weight_specialist.pth"
        )
        cls_sim_path_specialist = self.data_root.joinpath("cls_sim_specialist.pth")

        if not concept_feat_path_generalist.exists():
            raise RuntimeError(
                "need to call datamodule precompute_txt before using the model"
            )
        else:
            if select_idx_generalist is None:
                self.select_idx_generalist = th.load(select_idx_path_generalist)[
                    : cfg.num_concept
                ]
            else:
                self.select_idx_generalist = select_idx_generalist

            self.concepts_generalist = th.load(concept_feat_path_generalist)[
                self.select_idx_generalist
            ].cuda()
            if self.cfg.use_txt_norm:
                self.concepts_generalist = (
                    self.concepts_generalist
                    / self.concepts_generalist.norm(dim=-1, keepdim=True)
                )

            self.concept_raw_generalist = np.load(concept_raw_path_generalist)[
                self.select_idx_generalist
            ]
            self.concept2cls_generalist = (
                th.from_numpy(
                    np.load(concept2cls_path_generalist)[self.select_idx_generalist]
                )
                .long()
                .view(1, -1)
            )

        if not concept_feat_path_specialist.exists():
            raise RuntimeError(
                "need to call datamodule precompute_txt before using the model"
            )
        else:
            if select_idx_specialist is None:
                self.select_idx_specialist = th.load(select_idx_path_specialist)[
                    : cfg.num_concept
                ]
            else:
                self.select_idx_specialist = select_idx_specialist

            self.concepts_specialist = th.load(concept_feat_path_specialist)[
                self.select_idx_specialist
            ].cuda()
            if self.cfg.use_txt_norm:
                self.concepts_specialist = (
                    self.concepts_specialist
                    / self.concepts_specialist.norm(dim=-1, keepdim=True)
                )

            self.concept_raw_specialist = np.load(concept_raw_path_specialist)[
                self.select_idx_specialist
            ]
            self.concept2cls_specialist = (
                th.from_numpy(
                    np.load(concept2cls_path_specialist)[self.select_idx_specialist]
                )
                .long()
                .view(1, -1)
            )

        if init_weight_generalist is None:
            self.init_weight_concept_generalist(self.concept2cls_generalist)
        else:
            self.init_weight_generalist = init_weight_generalist
            
        if init_weight_specialist is None:
            self.init_weight_concept_specialist(self.concept2cls_specialist)
        else:
            self.init_weight_specialist = init_weight_specialist

        if (
            "cls_sim_prior" in self.cfg
            and self.cfg.cls_sim_prior
            and self.cfg.cls_sim_prior != "none"
        ):
            # class similarity is prior to restrict class-concept association
            # if class A and B are dissimilar (similarity=0), then the mask location will be 0
            print("use cls prior - initialising generalist")
            cls_sim_generalist = th.load(cls_sim_path_generalist)
            new_weights = []
            for concept_id in range(self.init_weight_generalist.shape[1]):
                target_class = int(
                    th.where(self.init_weight_generalist[:, concept_id] == 1)[0]
                )
                new_weights.append(
                    cls_sim_generalist[target_class]
                    + self.init_weight_generalist[:, concept_id]
                )
            self.init_weight_generalist = th.vstack(new_weights).T
            # self.weight_mask = cls_sim @ self.init_weight

        if (
            "cls_sim_prior" in self.cfg
            and self.cfg.cls_sim_prior
            and self.cfg.cls_sim_prior != "none"
        ):
            print("use cls prior - initialising specialist")
            cls_sim_specialist = th.load(cls_sim_path_specialist)
            new_weights = []
            for concept_id in range(self.init_weight_specialist.shape[1]):
                target_class = int(
                    th.where(self.init_weight_specialist[:, concept_id] == 1)[0]
                )
                new_weights.append(
                    cls_sim_specialist[target_class]
                    + self.init_weight_specialist[:, concept_id]
                )
            self.init_weight_specialist = th.vstack(new_weights).T

        self.asso_mat_generalist = th.nn.Parameter(self.init_weight_generalist.clone())
        self.asso_mat_specialist = th.nn.Parameter(self.init_weight_specialist.clone())

        print(f"Num labels is: {cfg.num_cls}")

        if "XRAY" in self.cfg.data_root:
            # we're doing multi-label classification
            self.train_acc = torchmetrics.Accuracy(
                num_labels=cfg.num_cls, task="multilabel"
            )
            self.valid_acc = torchmetrics.Accuracy(
                num_labels=cfg.num_cls, task="multilabel"
            )
            self.test_acc = torchmetrics.Accuracy(
                num_labels=cfg.num_cls, task="multilabel"
            )
        else:
            self.train_acc = torchmetrics.Accuracy(
                num_classes=cfg.num_cls, task="multiclass"
            )
            self.valid_acc = torchmetrics.Accuracy(
                num_classes=cfg.num_cls, task="multiclass"
            )
            # self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, average='macro')
            self.test_acc = torchmetrics.Accuracy(
                num_classes=cfg.num_cls, task="multiclass"
            )
        self.all_y = []
        self.all_pred = []
        self.confmat = torchmetrics.ConfusionMatrix(
            num_classes=self.cfg.num_cls, task="multiclass"
        )

        self.clip_model = self.cfg.clip_model

        # initialise the gating network
        self.gating_network = th.nn.Sequential(
            th.nn.Linear(cfg.img_feat_dim, cfg.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(cfg.hidden_dim, 1),
            th.nn.Sigmoid(),
        )

        self.save_hyperparameters()

    def _get_weight_mat_generalist(self):
        if self.cfg.asso_act == "relu":
            mat = F.relu(self.asso_mat_generalist)
        elif self.cfg.asso_act == "tanh":
            mat = F.tanh(self.asso_mat_generalist)
        elif self.cfg.asso_act == "softmax":
            mat = F.softmax(self.asso_mat_generalist, dim=-1)
        else:
            mat = self.asso_mat_generalist
        return mat

    def _get_weight_mat_specialist(self):
        if self.cfg.asso_act == "relu":
            mat = F.relu(self.asso_mat_specialist)
        elif self.cfg.asso_act == "tanh":
            mat = F.tanh(self.asso_mat_specialist)
        elif self.cfg.asso_act == "softmax":
            mat = F.softmax(self.asso_mat_specialist, dim=-1)
        else:
            mat = self.asso_mat_specialist
        return mat

    def forward_generalist(self, img_feat):
        mat = self._get_weight_mat_generalist()
        cls_feat = mat @ self.concepts_generalist
        sim = img_feat @ cls_feat.t()
        print(f"Sim: {sim}")
        return sim

    def forward_specialist(self, img_feat):
        mat = self._get_weight_mat_specialist()
        cls_feat = mat @ self.concepts_specialist
        sim = img_feat @ cls_feat.t()
        print(f"Sim: {sim}")
        return sim

    def forward(self, img_feat):
        generalist_sim = self.forward_generalist(img_feat)
        specialist_sim = self.forward_specialist(img_feat)

        gate = self.gating_network(img_feat).view(-1, 1)

        final_sim = gate * specialist_sim + (1 - gate) * generalist_sim
        return final_sim

    # def forward(self, img_feat):
    #     mat = self._get_weight_mat()
    #     cls_feat = mat @ self.concepts
    #     sim = img_feat @ cls_feat.t()
    #     th.set_printoptions(threshold=10000)
    #     print(f"Sim: {sim}")
    #     return sim

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        print(f"Data Root: {self.cfg.data_root}")

        sim = self.forward(image)
        pred = 100 * sim  # scaling as standard CLIP does
        # pred = sim

        # classification accuracy
        # TODO: update this for multi-label classification
        # print(pred.shape)
        # print(label.shape)

        # print(f"Pred: {pred}")
        # print(f"Label: {label}")

        if "XRAY" in self.cfg.data_root:
            # we need to do multi-label classification
            # print('doing binary cross entropy with logits')
            cls_loss = F.binary_cross_entropy_with_logits(pred, label.float())
        else:
            cls_loss = F.cross_entropy(pred, label)

        print(f"cls_loss: {cls_loss}")

        if th.isnan(cls_loss):
            import pdb

            pdb.set_trace()  # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()

        if self.cfg.asso_act == "softmax":
            row_l1_norm_generalist = th.linalg.vector_norm(
                F.softmax(self.asso_mat_generalist, dim=-1), ord=1, dim=-1
            ).mean()
            row_l1_norm_specialist = th.linalg.vector_norm(
                F.softmax(self.asso_mat_specialist, dim=-1), ord=1, dim=-1
            )
        # asso_mat sparse regulation
        row_l1_norm_generalist = th.linalg.vector_norm(
            self.asso_mat_generalist, ord=1, dim=-1
        ).max()  # .mean()
        row_l1_norm_specialist = th.linalg.vector_norm(
            self.asso_mat_specialist, ord=1, dim=-1
        )

        self.log("training_loss", cls_loss)
        self.log("mean l1 norm generalist", row_l1_norm_generalist)
        self.log("mean l1 norm specialist", row_l1_norm_specialist)
        self.log("div", div)

        # run sigmoid then threshold pred
        if "XRAY" in self.cfg.data_root:
            print("doing sigmoid then threshold")
            pred_label = (th.sigmoid(pred) > 0.5).float()
            print(pred_label)
        else:
            pred_label = pred

        self.train_acc(pred_label, label)
        print(f"train accuracy: {self.train_acc(pred_label, label)}")
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            # NOTE: this is not good - but we won't have to come here because use_l1_loss is set to False in the default config
            final_loss += self.cfg.lambda_l1 * (
                (row_l1_norm_generalist + row_l1_norm_specialist) / 2
            )
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def configure_optimizers(self):
        opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt

    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric("val_acc", summary="max")
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        if "XRAY" in self.cfg.data_root:
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
        else:
            loss = F.cross_entropy(pred, y)

        self.log("val_loss", loss)
        self.valid_acc(pred, y)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        print("running test step")
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        if "XRAY" in self.cfg.data_root:
            loss = F.binary_cross_entropy_with_logits(pred, y.float())
        else:
            loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log("test_loss", loss)
        self.test_acc(pred, y)
        print(f"test accuracy: {self.test_acc(pred, y)}")
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    # def test_epoch_end(self, outputs):
    #     all_y = th.hstack(self.all_y)
    #     all_pred = th.hstack(self.all_pred)
    #     self.total_test_acc = self.test_acc(all_pred, all_y)
    #     pass

    def on_test_epoch_end(self):
        all_y = th.hstack(self.all_y)
        all_pred = th.hstack(self.all_pred)
        self.total_test_acc = self.test_acc(all_pred, all_y)
        pass

    def on_predict_epoch_start(self):
        self.num_pred = 4
        self.concepts = self.concepts.to(self.device)

        self.pred_table = wandb.Table(
            columns=["img", "label"]
            + ["pred_{}".format(i) for i in range(self.num_pred)]
        )

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
        wandb.log({"pred_table": self.pred_table})

    def prune_asso_mat_generalist(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat_generalist().detach()
        val_asso_mat = th.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = th.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good

    def prune_ass_mat_specialist(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat_specialist().detach()
        val_asso_mat = th.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = th.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good

    def extract_cls2concept_generalist(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat_generalist().detach()
        strong_asso = asso_mat > thresh
        res = {}
        import pdb

        pdb.set_trace()
        for i, cls_name in enumerate(cls_names):
            ## threshold globally
            keep_idx = strong_asso[i]
            ## sort
            res[cls_name] = np.unique(self.concept_raw_generalist[keep_idx])
        return res

    def extract_cls2concept_specialist(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat_generalist().detach()
        strong_asso = asso_mat > thresh
        res = {}
        import pdb

        pdb.set_trace()
        for i, cls_name in enumerate(cls_names):
            keep_idx = strong_asso[i]
            res[cls_name] = np.unique(self.concept_raw_specialist[keep_idx])
        return res

    def extract_concept2cls_generalist(self, percent_thresh=0.95, mode="global"):
        asso_mat = self.asso_mat_generalist.detach()
        res = {}
        for i in range(asso_mat.shape[1]):
            res[i] = th.argsort(asso_mat[:, i], descending=True).tolist()
        return res

    def extract_concept2cls_specialist(self, percent_thresh=0.95, mode="global"):
        asso_mat = self.asso_mat_specialist.detach()
        res = {}
        for i in range(asso_mat.shape[1]):
            res[i] = th.argsort(asso_mat[:, i], descending=True).tolist()
        return res


class AssoConceptMoEFast(AssoConceptMoE):

    def forward_generalist(self, dot_product):
        print("running fast forward generalist")
        mat = self._get_weight_mat_generalist()
        res = dot_product @ mat.t()
        print(f"Res: {res}")
        return res

    def forward_specialist(self, dot_product):
        mat = self._get_weight_mat_specialist()
        res = dot_product @ mat.t()
        print(f"Res: {res}")
        return res

    def forward(self, dot_product):
        print("running fast")
        mat = self._get_weight_mat()
        res = dot_product @ mat.t()
        th.set_printoptions(threshold=10000)
        print(f"Res: {res}")
        return res

    # def forward(self, img_feat):
    #     generalist_sim = self.forward_generalist(img_feat)
    #     specialist_sim = self.forward_specialist(img_feat)

    #     gate = self.gating_network(img_feat).view(-1, 1)

    #     final_sim = gate * specialist_sim + (1 - gate) * generalist_sim
    #     return final_sim


# def AssoConceptMoE
