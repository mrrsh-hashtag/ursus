import os

log_file = os.environ.get("URSUS_LOG_FILE")
worker_host = os.environ.get("URSUS_WORKER_HOST")
worker_url = os.environ.get("URSUS_WORKER_URL")

mongo_url = os.environ.get("URSUS_MONGO_URL")
mongo_usr = os.environ.get("URSUS_MONGO_USER")
mongo_psw = os.environ.get("URSUS_MONGO_PSW")

var_list = [
    (log_file, "log_file"),
    (worker_host, "worker_host"),
    (worker_url, "worker_url"),
    (mongo_url, "mongo_url"),
    (mongo_usr, "mongo_usr"),
    (mongo_psw, "mongo_psw")
]
for var, var_str in var_list:
    if var is None:
        print(f"Warning: Environment variable {var_str} not found")
mongo_url = mongo_url.replace("<username>", mongo_usr).replace("<password>", mongo_psw)