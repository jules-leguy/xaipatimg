#%%
import os

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.3/"
test_dataset_filename = "bluediagonal_test.csv"

model_dir = os.environ["DATA"] + "models/db_v0.1.3/bluediagonal_model/"

#%%
# from xaipatimg.ml.xai import generate_cam_resnet18

# generate_cam_resnet18("gradcam", db_dir, test_dataset_filename, model_dir, "cuda:0")
#%%
from xaipatimg.ml.xai import generate_shap_resnet18

generate_shap_resnet18(db_dir, test_dataset_filename, model_dir, "cuda:1", n_jobs=32, dataset_size=1000)
#%%

#%%
