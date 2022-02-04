import os

log_file = os.environ.get("URSUS_LOG_FILE")
worker_host = os.environ.get("URSUS_WORKER_HOST")
worker_url = os.environ.get("URSUS_WORKER_URL")

mongo_url = os.environ.get("URSUS_MONGO_URL")
mongo_usr = os.environ.get("URSUS_MONGO_USER")
mongo_psw = os.environ.get("URSUS_MONGO_PSW")


for variable in [log_file, worker_host, worker_url, mongo_url, mongo_usr, mongo_psw]:
    if variable is None:
        print(f"Warning: Environment variable {variable} not found")
mongo_url = mongo_url.replace("<username>", mongo_usr).replace("<password>", mongo_psw)