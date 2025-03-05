import json
import uuid
import os

def generate_uuid():
    return str(uuid.uuid1().hex)

def load_db(db_dir):
    """
    Load the database from the given path. If the file does not exist, returns an empty DB.
    :param db_dir: path to the root directory of the database
    :return: database object
    """
    try:
        with open(os.path.join(db_dir, "db.json")) as json_file:
            json_data = json.load(json_file)
            return json_data
    except FileNotFoundError:
        return {}

def save_db(db_dir, db):
    """
    Save the given database to the given path
    :param db_dir: path to the root directory of the database
    :param db: database object
    :return: None
    """
    with open(os.path.join(db_dir, "db.json"), 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)
