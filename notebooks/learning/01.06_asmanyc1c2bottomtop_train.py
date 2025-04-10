#%%
import os

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.3/"
train_dataset_filename = "asmanyc1c2bottomtop_train.csv"
test_dataset_filename = "asmanyc1c2bottomtop_test.csv"
valid_dataset_filename = "asmanyc1c2bottomtop_valid.csv"
model_dir = os.environ["DATA"] + "models/db_v0.1.3/asmanyc1c2bottomtop_model/"

device = "cuda:1"
#%%
from xaipatimg.ml.learning import train_resnet18_model

train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir, device=device)
#%%
from xaipatimg.ml.learning import compute_resnet18_model_scores

compute_resnet18_model_scores(db_dir, train_dataset_filename, test_dataset_filename, valid_dataset_filename, model_dir, device=device)
#%%
