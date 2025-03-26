_base_ = 'HAM10000_base.py'
n_shots = 2
data_root = 'exp/asso_opt/HAM10000_generalist/HAM10000_generalist_2shot_fac'

lr = 1e-3
bs = 4
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]