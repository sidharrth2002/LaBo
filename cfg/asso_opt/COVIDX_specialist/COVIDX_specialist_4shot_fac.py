_base_ = 'COVIDX_base.py'
n_shots = 4
data_root = 'exp/asso_opt/COVIDX_specialist/COVIDX_specialist_4shot_fac'
init_val = 0.01

lr = 1e-4
bs = 8
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 1]