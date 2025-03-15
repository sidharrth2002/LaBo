_base_ = 'XRAY_NIH_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/XRAY_NIH/XRAY_NIH_allshot_fac'
lr = 5e-4
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]