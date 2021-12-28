import sys
import logging
from ursus import custom_logger
import ursus.mongo_connection as mongo
import ursus.worker as worker
import config as cfg
import pika

custom_logger.setup()
log = logging.getLogger(__name__)


def check_mongo():
    log.info("Checking MongoDB connection")
    if mongo.connect() is not None:
        sys.exit()
    log.info("MongoDB connection OK")


def start_worker():
    log.info("Starting Ursus worker")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=cfg.worker_host)
    )
    queue = "ursus_task_queue"
    channel = connection.channel()
    channel.queue_declare(queue=queue)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue,
                          on_message_callback=worker.ursus_worker)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print()
        log.info("Stopping Ursus worker")

def test_worker():
    body = {
        "job_id": "test",
        "customer": "Default Company",
        "frame_name": "DC Revenue",
        "features": {
            "dep_var": "revenue",
            "date_var": "DATE",
            "ind_var": [
                "tv_S",
                "ooh_S",
                "print_S",
                "facebook_I",
                "search_clicks_P",
                "search_S",
                "competitor_sales_B",
                "facebook_S"
            ]
        }
    }
    worker.ursus_worker("", "", "", body)

if __name__ == "__main__":
    check_mongo()
    start_worker()