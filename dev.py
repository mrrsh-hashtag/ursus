import click

from den.ursus import Ursus
from development import cases
from development.reporter import Reporter
from utilities import custom_logger


@click.command()
@click.argument("data")
@click.option('--quick', '-q', is_flag=True, help="Very low budget.")
@click.option('--short', '-s', is_flag=True, help="Medium budget.")
@click.option('--ws', is_flag=True, help="Workshop mode.")
def run(data, quick, short, ws):
    """
    Run Ursus on Demo Data
    """
    data_path, params = cases.get(data)
    if quick:
        params["budget"] = 60
        params["trials"] = 2

    elif short:
        params["budget"] = 500
    
    ursus = Ursus(data_path, params)
    if ws:
        ursus.workshop_test()
        return
    else:
        ursus.train()
        Reporter(ursus)


if __name__ == "__main__":
    custom_logger.setup()
    run()
