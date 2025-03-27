steps = 8
n_runs = 1
img_path = 'datasets/XRAY_NIH/images/'
num_cls = 14
unfreeze_clip = False
paper = True
data_root = 'exp/linear_probe/XRAY_NIH'
img_split_path = 'datasets/XRAY_NIH/splits'
cls_names = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
img_ext = ''
clip_model = 'ViT-B/16'
lr = 0.001
bs = 128
n_shots = 'all'
dataset = 'XRAY_NIH'
