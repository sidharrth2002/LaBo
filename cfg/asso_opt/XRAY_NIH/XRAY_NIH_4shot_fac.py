_base_ = 'XRAY_NIH_base.py'
n_shots = 4
data_root = 'exp/asso_opt/XRAY_NIH/XRAY_NIH_4shot_fac'
init_val = 0.01

lr = 1e-4
bs = 8

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 1]