from ursus.ursus import Ursus
from ursus import custom_logger
import ursus.mongo_connection as mongo

custom_logger.setup()
params = {
    "customer": "Default Company",
    "frame_name": "DC Revenue",
    "dep_var": "revenue",
    "date_var": "DATE",
    "ind_var": [
        "tv_S",
        "ooh_S",
        "print_S",
        "search_clicks_P",
        "search_S",
        "facebook_S"
    ],
    "context": [
        "facebook_I",
        "competitor_sales_B",
    ],
    "budget": 50,
    "print_progress": True,
    # "plot_one_page": True
    # "country_code": "DE"
}

ursus = Ursus("demo_data.csv", params)
ursus.train()

training_data = ursus.get_training_data()
# mongo.connect()
# mongo.set_training_results(params["customer"], params["frame_name"], training_data)