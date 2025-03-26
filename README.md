# xaipatimg

Contains a set of tools to generate datasets of images which are grids of geometrical shapes, and to extract subsets of
images that follow certain patterns. Also contains tools to facilitate the learning on the generated datasets with 
pytorch.

## Database generation

A database is meant to be generated once and contain a large amount of images, from which training datasets will be 
extracted according to rules. A dataset is composed of a folder that contains all the images, and a json file that index
their paths and content. A dataset is identified by the path of its root folder.

```python
db_dir = "path/to/database_v1.0"
```   
### Content of the database

To generate the images, the user must specify their content represented in a structured format (python dict or json 
file).

An existing database can be loaded as such.

```python
from xaipatimg.datagen.dbimg import load_db

db = load_db(db_dir)    
```

Or it can be created from scratch.

```python
db = {}
```

Then it must be filled according to the following format :
 
```python
{
    "unique_id_1" : {                          # unique id of the image in the DB
        "path" : "relative/path/to/img1.jpg",  # path where the image will be generated
        "division" : (16, 16),                 # divition of the grid
        "size" : (1000, 1000),                 # size of the generated image in pixels
        "content" : [                          # list of the geometrical shapes contained in the image
            {
            "pos" : (x1, y1),                  # coordinates in the grid
            "shape": "square",                 # shape (among square/triangle/circle)
            "color : "#HEXCODE"                # hex code of the color
            },
            ...,
            {
            "pos" : (xn, yn),
            "shape": "circle",
            "color : "#HEXCODE"
            },
        ]
    },
    "unique_id_2" : {
        "path" : "relative/path/to/img2.jpg",
        "division" : (16, 16),
        "size" : (1000, 1000),
        "content" : [
            {
            "pos" : (x1, y1),
            "shape": "triangle",
            "color : "#HEXCODE"
            },
            ...,
            {
            "pos" : (xn, yn),
            "shape": "triangle",
            "color : "#HEXCODE"
            },
        ]
    } 
}
```

### Generation of the images

Once the database content is set, the images can be generated using the following function.

```python
from xaipatimg.datagen.genimg import gen_img_and_save_db

gen_img_and_save_db(db,  # database dict object 
                    db_dir,  # path to the root folder
                    overwrite=False,  # whether to overwrite images that may already exist in the database folder
                    n_jobs=1)  # number of jobs for parallel execution
```

## Dataset extraction

A training dataset can be extracted from a database, based on a rule that defines whether an instance is of positive or 
negative class.

Let's define a simple rule that considers an image as positive if and only if it contains at least one square.

```python
def contains_square(img_content):
    for c in img_content:
        if c["shape"] == "square":
            return True
    return False
```

Then a learning dataset for this rule can be created the following way.

```python
from xaipatimg.datagen.gendataset import create_dataset_based_on_rule
import os

csv_name_train = "contains_square_train.csv"
csv_name_valid = "contains_square_valid.csv"

create_dataset_based_on_rule(db_dir,          # path to the directory that contains the DB
                             csv_name_train,  # name of the csv file that contains the training dataset
                             csv_name_train,  # name of the csv file that contains the training dataset
                             csv_name_valid,  # name of the csv file that contains the validation dataset
                             valid_size=0.2,  # proportion or number of samples in the validation dataset    
                             dataset_pos_samples_nb=10000,
                             # number of positive samples to be extracted from the database
                             dataset_neg_samples_nb=10000,
                             # number of negative samples to be extracted from the database
                             rule_fun=contains_square)  # function that defines the rule
```

In order to visualize a sample of the dataset, the following function can be used.

```python
from xaipatimg.datagen.gendataset import extract_sample_from_dataset

extract_sample_from_dataset(db_dir,  # path to the folder that contains the DB
                            csv_name_train,  # path to the csv file that contains the dataset
                            output_dir_path=sample_img_path,  # name of the folder in which the sample will be written
                            pos_samples_nb=1000,  # number of positive samples
                            neg_samples_nb=1000)  # number of negative samples
```
