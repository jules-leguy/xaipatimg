#%%
import os

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.3/"
train_dataset_filename = "bluediagonal_train.csv"
valid_dataset_filename = "bluediagonal_valid.csv"
model_dir = os.environ["DATA"] + "models/db_v0.1.3/bluediagonal_model/"
#%% md
# 
#%%
from xaipatimg.ml.learning import train_resnet18_model

train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir)
#%%

#%%
