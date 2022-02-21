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
        "search_S",
        "facebook_S"
    ],
    "context": [
        "search_clicks_P",
        "facebook_I",
        "competitor_sales_B",
    ],
    "budget": 2000,
    # "auto_weight_season": True,
    "print_progress": True,
    "plot_one_page": True,
    # "adstock_algo": "weibull",
    "adstock_algo": "geometric",
    # "country_code": "DE"
}

ursus = Ursus("demo_data.csv", params)
# ursus.workshop_test()
ursus.train()
# ursus.get_budget_optimization()

training_data = ursus.get_training_data()
do_push = input("Push training data [y/n]:")
if do_push == "y" or do_push == "Y":
    mongo.connect()
    mongo.set_training_results(params["customer"], params["frame_name"], training_data)