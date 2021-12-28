from ursus.ursus import Ursus
params = {
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

ursus = Ursus("demo_data.csv", params)
ursus.train()
