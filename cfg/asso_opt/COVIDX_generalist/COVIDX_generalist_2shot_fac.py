_base_ = 'COVIDX_base.py'
n_shots = 2
data_root = 'exp/asso_opt/COVIDX_generalist/COVIDX_generalist_2shot_fac'

lr = 1e-3
bs = 4
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]