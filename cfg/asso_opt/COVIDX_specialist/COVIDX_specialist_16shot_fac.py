_base_ = 'COVIDX_base.py'
n_shots = 16
data_root = 'exp/asso_opt/COVIDX_specialist/COVIDX_specialist_16shot_fac'
init_val = 0.1

lr = 1e-3
bs = 16
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 15]