import traceback
from datetime import datetime
import sys
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import config as cfg
import logging
import pandas as pd
log = logging.getLogger(__name__)
from pprint import pprint

client = None


def connect():
    global client
    try:
        client = MongoClient(cfg.mongo_url,
                             serverSelectionTimeoutMS=3000)
        client.server_info()
    except ServerSelectionTimeoutError as err:
        traceback.print_exception(*sys.exc_info())
        log.critical(f"Unable to connect to MongoDB at {cfg.mongo_url}")
        return 1

def get_dataframe(customer, frame_name):
    document = client.ursus_data[customer]
    df = document.find_one({"name": frame_name}, {"data_frame": 1})
    try:
        df = pd.DataFrame.from_records(df["data_frame"])
    except TypeError:
        log.error(f"Dataframe '{frame_name}' not found in customer '{customer}'")
        raise AttributeError
    return df

def set_training_results(customer, frame_name, training_data):
    document = client.ursus_data[customer]
    if not "timestamp" in training_data:
        training_data["timestamp"] = datetime.now()
    document.update_many(
    {
        "name": frame_name
    },
    {
        "$set": {
            "training_data": training_data
        }
    }
)