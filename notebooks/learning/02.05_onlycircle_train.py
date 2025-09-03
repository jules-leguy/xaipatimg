import os

os.environ["DATA"] = os.path.expanduser("~/")

db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.5_6x6/"
train_dataset_filename = "onlycircle_train.csv"
test_dataset_filename = "onlycircle_test.csv"
valid_dataset_filename = "onlycircle_valid.csv"
model_dir = os.environ["DATA"] + "models/db0.1.5_6x6/onlycircle_90accuracy_model/"
classification_dir = os.environ["DATA"] + "classification/db0.1.5_6x6/"

device = "cuda:0"

from xaipatimg.ml.learning import train_resnet18_model

train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir, target_accuracy=0.9, device=device)

from xaipatimg.ml.learning import compute_resnet18_model_scores

compute_resnet18_model_scores(db_dir, train_dataset_filename, test_dataset_filename, valid_dataset_filename, model_dir, device=device)

from xaipatimg.ml.learning import save_classification

save_classification(db_dir=db_dir, test_dataset_filename=test_dataset_filename, model_dir = model_dir, classification_dir=classification_dir, device=device)
print("TP/TN/FP/FN saved")