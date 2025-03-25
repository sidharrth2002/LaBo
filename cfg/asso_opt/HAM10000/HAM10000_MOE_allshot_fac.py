_base_ = 'HAM10000_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/HAM10000/HAM10000_MOE_allshot_fac'
lr = 5e-4
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]

proj_name = "HAM10000"
concept_root = 'datasets/HAM10000/concepts_generalist/'
img_split_path = 'datasets/HAM10000/splits'
img_path = 'datasets/HAM10000/images'

raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 7

img_feat_dim = 512
hidden_dim = 256

# run mixture of experts
model_type="moe"

clip_model = 'ViT-B/16'
