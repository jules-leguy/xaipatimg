import os
os.environ["DATA"] = os.path.expanduser("~/")

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.4_6x6/"
test_dataset_filename = "bluediagonal_test.csv"
model_dir = os.environ["DATA"] + "models/db0.1.4_6x6/bluediagonal_80accuracy_model/"

from xaipatimg.ml.xai import generate_cam_resnet18

generate_cam_resnet18("gradcam", db_dir, test_dataset_filename, model_dir, "cuda:0")

import numpy as np
from xaipatimg.ml.xai import generate_shap_resnet18

generate_shap_resnet18(db_dir, test_dataset_filename, model_dir, "cuda:1", n_jobs=20, dataset_size=500, masker="ndarray")

