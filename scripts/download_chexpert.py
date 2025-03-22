"""
Download only the validation set of the CheXpert dataset.
"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashery/chexpert")

print("Path to dataset files:", path)