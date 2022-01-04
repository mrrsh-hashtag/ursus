from ursus.ursus import Ursus
from ursus import custom_logger
custom_logger.setup()
params = {
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
        "budget": 5000,
        # "country_code": "DE"
    }

ursus = Ursus("demo_data.csv", params)
ursus.train()

