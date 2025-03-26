#%%
import os

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.3/"
train_dataset_filename = "twiceasmanyredasblue_train.csv"
test_dataset_filename = "twiceasmanyredasblue_test.csv"
valid_dataset_filename = "twiceasmanyredasblue_valid.csv"
model_dir = os.environ["DATA"] + "models/db_v0.1.3/twiceasmanyredasblue_model/"

device = "cuda:0"
#%%
from xaipatimg.ml.learning import train_resnet18_model

train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir)
#%%
from xaipatimg.ml.learning import compute_resnet18_model_scores

compute_resnet18_model_scores(db_dir, train_dataset_filename, test_dataset_filename, valid_dataset_filename, model_dir, device=device)
#%%
