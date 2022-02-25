from os.path import abspath, dirname, join

case_dd = {
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
    "budget": 2000,
    "trials": 3,
    "auto_weight_season": True,
    "print_progress": True,
    # "plot_one_page": True,
    "adstock_algo": "weibull",
    # "adstock_algo": "geometric",
    # "country_code": "DE"
}

case_kk = {
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

case_lv = {
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

main_path = join(dirname(dirname(abspath(__file__))), "data")

case_dict = {
    "dd": (
        join(main_path, "demo_data.csv"),
        case_dd
    ),
    "kk": (
        join(main_path, "KK_agg.csv"),
        case_kk
    ),
    "lv": (
        join(main_path, "LV.csv"),
        case_lv
    )
}

def get(case):
    if not case in case_dict:
        raise AttributeError(f"Case {case} not in dict")
    return case_dict[case]