_base_ = 'HAM10000_base.py'
n_shots = 8
data_root = 'exp/asso_opt/HAM10000_specialist/HAM10000_specialist_8shot_fac'
init_val = 0.01

lr = 1e-3
bs = 8
cls_name_init = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 10]