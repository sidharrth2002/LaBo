_base_ = 'COVIDX_base.py'
n_shots = 1
data_root = 'exp/asso_opt/COVIDX_specialist/COVIDX_specialist_1shot_fac'

lr = 1e-3
bs = 4

max_epochs = 10000
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]