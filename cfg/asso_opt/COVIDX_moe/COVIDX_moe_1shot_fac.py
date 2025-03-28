_base_ = 'COVIDX_base.py'
n_shots = 1
data_root = 'exp/asso_opt/COVIDX_moe/COVIDX_moe_1shot_fac'

init_weight_path_generalist = data_root + "/init_weight.pth"
init_weight_path_specialist = data_root + "/init_weight_specialist.pth"

lr = 1e-3
bs = 4

max_epochs = 10000
cls_name_init = 'yes'
# warm_start_training = 'yes'

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]