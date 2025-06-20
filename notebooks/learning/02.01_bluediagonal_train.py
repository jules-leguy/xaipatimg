import os

os.environ["DATA"] = os.path.expanduser("~/")

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.4_6x6/"
train_dataset_filename = "bluediagonal_train.csv"
test_dataset_filename = "bluediagonal_test.csv"
valid_dataset_filename = "bluediagonal_valid.csv"
model_dir = os.environ["DATA"] + "models/db0.1.4_6x6/bluediagonal_80accuracy_model/"

device = "cuda:0"

from xaipatimg.ml.learning import train_resnet18_model

train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir, device=device)

from xaipatimg.ml.learning import compute_resnet18_model_scores

compute_resnet18_model_scores(db_dir, train_dataset_filename, test_dataset_filename, valid_dataset_filename, model_dir, device=device)