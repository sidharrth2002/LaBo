_base_ = 'XRAY_NIH_base.py'
n_shots = 16
data_root = 'exp/asso_opt/XRAY_NIH/XRAY_NIH_16shot_fac'
init_val = 0.1

lr = 1e-3
bs = 16

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 15]