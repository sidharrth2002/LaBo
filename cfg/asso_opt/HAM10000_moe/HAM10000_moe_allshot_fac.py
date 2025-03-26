_base_ = 'HAM10000_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/HAM10000_moe/HAM10000_moe_allshot_fac'
lr = 5e-4
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]

