import json
import logging

from pymongo.common import validate

import ursus.mongo_connection as mongo
from ursus.ursus import Ursus

log = logging.getLogger(__name__)

def ursus_worker(ch, method, properties, params):
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if not isinstance(params, dict):
        params = json.loads(params.decode())
    invalid = valid_task(params)
    if invalid:
        log.error(f"Invalid task description. {invalid}")
        return
    log.info(f"Received job {params['job_id']} from {params['customer']} with {params['frame_name']}")
    try:
        df = mongo.get_dataframe(params["customer"], params["frame_name"])
    except AttributeError:
        log.warn("Worker stops. No dataframe found")
        return
    
    log.info(f"{params['job_id']} Starting training")
    ursus = Ursus(df, params["features"])
    ursus.train(plot=False)
    log.info(f"{params['job_id']} Extracting training data")
    training_data = ursus.get_training_data()
    mongo.set_training_results(params["customer"], params["frame_name"], training_data)

    # jdb = get_json()
    # jdb[params["company"]][params["table_id"]]["ursus_data"] = ursus_json_data
    # set_json(jdb)
    log.info(f"{params['job_id']} Worker returning ack")
    

def valid_task(params):
    keys = ["customer", "frame_name", "features", "job_id"]
    missing_keys = [key for key in keys if key not in params.keys()]
    if missing_keys:
        return f"Missing keys: " + str(missing_keys)
    else:
        return False
