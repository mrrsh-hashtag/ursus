from den.ursus import Ursus
from utilities import custom_logger
import utilities.mongo_connection as mongo

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
    "thetas": {
        "tv_S": {
            "lower": 0.4,
            "upper": 0.99
        },
        "ooh_S": {
            "lower": 0.4,
            "upper": 0.99
        },
        "search_S": {
            "lower": 0.0,
            "upper": 0.5
        },
        "facebook_S": {
            "lower": 0.0,
            "upper": 0.5
        },
    },
    "taus": {
        "tv_S": {
            "lower": 4,
        },
        "ooh_S": {
            "lower": 4,
        },
        "search_S": {
            "lower": 0.5,
            "upper": 2
        },
        "facebook_S": {
            "lower": 0.5,
            "upper": 2
        },
    },
    "budget": 50,
    "trials": 1,
    # "budget": 500,
    "auto_weight_season": True,
    "print_progress": True,
    "plot_one_page": True,
    "adstock_algo": "weibull",
    # "adstock_algo": "geometric",
    # "country_code": "DE"
}

params_KK = {
    "customer": "",
    "frame_name": "",
    "dep_var": "Conversions",
    "date_var": "Date",
    "ind_var": [
        "TV",
        "OLV",
        "OOH",
        "Radio",
        "Display",
        "Mail",
        "Influencers",
        "Print",
        "PaidSocial",
        "Search"

    ],
    "context": [
        "Elpris",
        "BankID"
    ],
    "thetas": {
        "tv_S": {
            "lower": 0.4,
            "upper": 0.99
        },
        "ooh_S": {
            "lower": 0.4,
            "upper": 0.99
        },
        "search_S": {
            "lower": 0.0,
            "upper": 0.5
        },
        "facebook_S": {
            "lower": 0.0,
            "upper": 0.5
        },
    },
    "taus": {
        "tv_S": {
            "lower": 4,
        },
        "ooh_S": {
            "lower": 4,
        },
        "search_S": {
            "lower": 0.5,
            "upper": 2
        },
        "facebook_S": {
            "lower": 0.5,
            "upper": 2
        },
    },
    "budget": 5000,
    # "budget": 500,
    "auto_weight_season": True,
    "print_progress": True,
    "plot_one_page": True,
    # "adstock_algo": "weibull",
    "adstock_algo": "geometric",
    # "country_code": "DE"
}

params_LV = {
    "customer": "Demo Company",
    "frame_name": "New Deposit Customers",
    "dep_var": "NDC",
    "date_var": "date",
    "ind_var": [
        "apple_app_store_EUR",
        "display_direct_EUR",
        "display_programmatic_EUR",
        "facebook_app_EUR",
        "facebook_web_EUR",
        "google_play_store_EUR",
        "google_search_branded_EUR",
        "google_search_furtive_EUR",
        "google_search_generic_EUR",
        "radio_EUR",
        "snapchat_app_EUR",
        "tv_EUR",
        "youtube_EUR"
    ],
    "context": [
        "competition_TRP"
    ],
    
    # "budget": 2000,
    "budget": 60,
    # "budget": 500,
    # "auto_weight_season": True,
    "print_progress": True,
    "plot_one_page": True,
    # "adstock_algo": "weibull",
    "adstock_algo": "geometric",
    # "country_code": "DE"
    "trials": 5
}


ursus = Ursus("data/demo_data.csv", params)
# ursus = Ursus("KK_agg.csv", params_KK)
# ursus = Ursus("LV.csv", params_LV)

# ursus.workshop_test()
ursus.train()
# ursus.get_budget_optimization()

# while True:
#     print("Low risk")
#     ursus.get_budget_optimization()
#     print("High risk")
#     ursus.get_budget_optimization(0.6, 1.4)
#     do_retry = input("Generate new opti [y/n]:")
#     if do_retry == "y" or do_retry == "Y":
#         ursus.budget_optimizations = []
#     else:
#         break    

# training_data = ursus.get_training_data()
# do_push = input("Push training data [y/n]:")
# if do_push == "y" or do_push == "Y":
#     mongo.connect()
#     mongo.set_training_results(params["customer"], params["frame_name"], training_data)