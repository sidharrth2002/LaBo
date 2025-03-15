_base_ = 'XRAY_NIH_base.py'
n_shots = 2
data_root = 'exp/asso_opt/XRAY_NIH/XRAY_NIH_2shot_fac'

lr = 1e-3
bs = 4

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]