import json
import uuid
import os

def generate_uuid():
    return str(uuid.uuid1().hex)

def load_db(db_root_path):
    """
    Load the database from the given path. If the file does not exist, returns an empty DB.
    :param db_root_path: path to the root of the database
    :return: database object
    """
    try:
        with open(os.path.join(db_root_path, "db.json")) as json_file:
            json_data = json.load(json_file)
            return json_data
    except FileNotFoundError:
        return {}

def save_db(db_root_path, db):
    """
    Save the given database to the given path
    :param db_root_path: path to the root of the database
    :param db: database object
    :return: None
    """
    with open(os.path.join(db_root_path, "db.json"), 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

"""
Database format
{
    "hash1" : {
        "path" : "path/to/img1.jpg",
        "division" : (16, 16),
        "size" : (1000, 1000),
        "content" : [
            {
            "pos" : (x1, y1),
            "shape": "triangle/square/circle",
            "color : "#HEXCODE"
            },
            ...,
            {
            "pos" : (xn, yn),
            "shape": "triangle/square/circle",
            "color : "#HEXCODE"
            },
        ]
    },
    "hash2" : {
        "path" : "path/to/img2.jpg",
        "division" : (16, 16),
        "size" : (1000, 1000),
        "content" : [
            {
            "pos" : (x2, y2),
            "shape": "triangle/square/circle",
            "color : "#HEXCODE"
            },
            ...,
            {
            "pos" : (xn, yn),
            "shape": "triangle/square/circle",
            "color : "#HEXCODE"
            },
        ]
    }
    
}
"""