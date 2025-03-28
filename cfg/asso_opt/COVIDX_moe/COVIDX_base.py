_base_ = '../base.py'
# dataset 
proj_name = "COVIDX_moe"
concept_root = 'datasets/COVIDX/concepts_generalist/'
img_split_path = 'datasets/COVIDX/splits'
img_path = 'datasets/COVIDX/images'

model_type = 'moe'
img_feat_dim = 512
hidden_dim = 256

concept_type = "all"
img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 4

## data loader
bs = 32
on_gpu = True

# concept select
num_concept = num_cls * 50

# weight matrix fitting
lr = 1e-4
max_epochs = 5000
# max_epochs=100

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
clip_model = 'ViT-B/16'
# clip_model = 'medclip'
# clip_model = 'biomedclip'
