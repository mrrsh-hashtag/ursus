import os
from datetime import datetime
from os.path import abspath, dirname, join
from pathlib import Path
import pprint

import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Reporter:
    N_COLS = 2
    TEMPLATE="plotly_dark"
    def __init__(self, bear_model):
        self.bear = bear_model
        self.rows = [[]]
        self.build_cells()
        self.compile_report()
    
    def build_cells(self):

        cells = [
            self.plot_regression,
            self.plot_errors,
            self.plot_pareto,
            self.plot_training_eval,
            self.plot_spend_effect,
            self.plot_waterfall,
            self.plot_budget_optimizer,
            self.plot_adstock,
            self.print_hyper_params,
            self.print_config_params,
        ]
        for cell in cells:
            cell_str = cell()
            self.add_cell(cell_str)

    def plot_regression(self):
        data = self.bear.get_regression_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["x_date"],
            y=data["y_true"],
            mode="lines",
            name="y true"))
        fig.add_trace(go.Scatter(
            x=data["x_date"],
            y=data["y_pred"],
            mode="lines",
            name="y pred"))
        fig.add_trace(go.Scatter(
            x=data["x_date"],
            y=data["y_season"],
            mode="lines",
            name="season"))
        fig.update_layout(
            title_text=f"Regression. r2: {data['r2']:.4f}",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_errors(self):
        data = self.bear.get_regression_data()
        y = data["y_true"] - data["y_pred"]
        y_std = np.std(y)
        std1_p = np.ones_like(y) * (np.mean(y) + y_std)
        std1_n = np.ones_like(y) * (np.mean(y) - y_std)

        fig = go.Figure()
        # Plot stds
        fig.add_trace(go.Scatter(
            x=np.concatenate([data["x_date"], data["x_date"][::-1]]),
            y=np.concatenate([std1_p, std1_n[::-1]]),
            fill='toself',
            hoveron='points',
            name="STD 1"
        ))
        
        # Plot error curve
        fig.add_trace(go.Scatter(
            x=data["x_date"],
            y=y,
            mode="lines",
            name="Error"))
        fig.update_layout(
            title_text="Error plot",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_pareto(self):
        data = self.bear.get_pareto_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["rmse"],
            y=data["rssd"],
            opacity=0.5,
            marker=dict(
                size=8,
                cmax=data["c_vals"][-1],
                cmin=0,
                color=data["c_vals"],
                colorbar=dict(
                    title="Colorbar"
                ),
                colorscale="Turbo"
            ),
            mode="markers"))
        fig.update_layout(
            title_text=f"Pareto front. Loss: {data['loss']:.4f}",
            yaxis_range=[0,1],
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_training_eval(self):
        data = self.bear.get_training_eval_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["r2"],
            mode="lines",
            name="r2"))
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["rmse_rssd"],
            mode="lines",
            name="rmse_rssd"))
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["rmse"],
            mode="lines",
            name="rmse"))
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["rssd"],
            mode="lines",
            name="rssd"))
        fig.update_layout(
            title_text=f"Training evaluation.",
            yaxis_range=[0,1.2],
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_spend_effect(self):
        data = self.bear.get_spend_effect_data()
        fig = go.Figure(data=[
            go.Bar(name="Share of Spend", x=data["labels"], y=data["spend_share"]),
            go.Bar(name="Share of Effect", x=data["labels"], y=data["effect_share"])
        ])
        fig.update_layout(
            title_text=f"Share of Spend vs Effect.",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_waterfall(self):
        data = self.bear.get_waterfall_data()
        fig = go.Figure(go.Waterfall(
            x=data["amount"].values,
            y=data.index,
            name="20",
            orientation="h",
            textposition="outside",
        ))
        fig.update_layout(
            title_text=f"Waterfall.",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)
    
    def plot_budget_optimizer(self):
        opt_obj = self.bear.get_budget_optimization(get_model=True)
        df = opt_obj.get_df()
        df = df / df.sum()
        
        data = []
        for i, col in enumerate(df.columns):
            y_ax, osg = "y", i + 1
            if "effect" in col:
                y_ax = "y2"
            data.append(
                go.Bar(name=col.title(),
                x=df.index,
                y=df[col].values,
                yaxis=y_ax,
                offsetgroup=osg)
            )

        fig = go.Figure(
            data=data,
            layout={
                "yaxis": {"title": "Spend"},
                "yaxis2": {
                    "title": "Effect",
                    "overlaying": "y",
                    "side": "right"
                }
            }
        )
        title = " ".join([
            "Budget opimization"
            f"Budget change {opt_obj.budget_change*100:.1f}%",
            f"Effect change {opt_obj.effect_change*100:.1f}%"
        ])
        fig.update_layout(
            title_text=title,
            barmode="group",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def plot_adstock(self):
        channel_weights = self.bear.get_adstock_data()
        fig = go.Figure()
        for channel, weights in channel_weights.items():
            if channel == "time_lag":
                continue
            name = channel.replace("_EUR", "").replace("google", "ggl")\
                .replace("facebook", "fb")
            fig.add_trace(go.Scatter(
                x=channel_weights["time_lag"],
                y=weights,
                mode="lines",
                name=name))
        fig.update_layout(
            title_text="Adstock",
            template=self.TEMPLATE)
        return fig.to_html(full_html=False)

    def print_hyper_params(self):
        data = {}
        for channel_param, value in self.bear.hyper_params.items():
            ch_split = channel_param.split("_")
            channel, param = "_".join(ch_split[0:-1]), ch_split[-1]
            if not channel in data:
                data[channel] = {param: value}
            else:
                data[channel][param] = value
        df = pd.DataFrame.from_dict(data).T
        return "<h3>Hyper Parameters</h3>" + df.to_html()

    def print_config_params(self):
        params = {}
        for k, v in self.bear.parameters.items():
            if isinstance(v, (dict, list)):
                params[k] = str(v)
            else:
                params[k] = v
        df = pd.DataFrame(params, index=[0]).T
        return "<h3>Parameters</h3>" + df.to_html(header=False)


    def add_cell(self, cell_str):
        cell = f'<div class="column">{cell_str}</div>'
        if len(self.rows[-1]) < self.N_COLS:
            self.rows[-1].append(cell)
        else:
            self.rows.append([cell])

    def get_out_path(self):
        now = datetime.now().strftime("%Y %m %d %H:%M:%S")
        base_dir = dirname(dirname(abspath(__file__)))
        repo_dir = join(base_dir, "data", "reports")
        Path(repo_dir).mkdir(parents=True, exist_ok=True)
        return join(repo_dir, now + ".html")

    def get_style(self):
        return """
        <style>
        * {
        box-sizing: border-box;
        }

        /* Create two equal columns that floats next to each other */
        .column {
        float: left;
        width: 50%;
        padding: 10px;

        }

        /* Clear floats after the columns */
        .row:after {
        content: "";
        display: table;
        clear: both;
        }
        table {
        font-family: arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        }

        td, th {
        border: 1px solid #aaaaaa;
        text-align: left;
        padding: 8px;
        }

        tr:nth-child(even) {
        background-color: #aaaaaa;
        }
        tr:nth-child(odd) {
        color:gray
        }
        h3 {
            font-family: "Open Sans",verdana,arial,sans-serif;
            color:gray

        }
        </style>"""
    
    def get_div_rows(self):
        html_str = ""
        for row in self.rows:
            cols = "\n".join(row)
            html_str += f'<div class="row">{cols}</div>'
        return html_str

    def get_template(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        {style}
        <body style="background-color:rgb(25, 35, 25);">
        {div_rows}
        </body>
        </html>
        """

    def compile_report(self):
        report_str = self.get_template()
        style = self.get_style()
        div_rows = self.get_div_rows()
        report_str = report_str.format(style=style, div_rows=div_rows)
        self.save_html(report_str)
    
    def save_html(self, html_str):
        out_path = self.get_out_path()
        with open(out_path, "w") as fp:
            fp.write(html_str)
        os.system(f"open '{out_path}'")





def table_html(headers,data):
    pre_existing_template="<html>"+"<head>"+"<style>"
    pre_existing_template+="table, th, td {border: 1px solid black;border-collapse: collapse;border-spacing:8px}"
    pre_existing_template+="</style>"+"</head>"
    pre_existing_template+="<table style='width:50%'>"
    pre_existing_template+='<tr>'
    for header_name in headers:
        pre_existing_template+="<th style='background-color:#3DBBDB;width:85;color:white'>"+header_name+"</th>"
    pre_existing_template+="</tr>"
    for i in range(len(data[0])):
        sub_template="<tr style='text-align:center'>"
        for j in range(len(headers)):
            sub_template+="<td>"+str(data[j][i])+"</td>"
        sub_template+="<tr/>"
        pre_existing_template+=sub_template
    pre_existing_template+="</table>"
    return(pre_existing_template)

def first_plot():
    x = np.linspace(0, 6)
    y = np.sin(x) + np.random.normal()
    xy_data = go.Scatter( x=x, y=y, mode='markers', marker=dict(size=4), name='Season')
    data = [xy_data]
    first_plot_url = py.plot(data, filename='apple stock moving average', auto_open=False)

    return first_plot_url


def insert_html(plot1):
    return '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
    <body>
        <h1>2014 technology and CPG stock prices</h1>

        <!-- *** Section 1 *** --->
        <h2>Section 1: Apple Inc. (AAPL) stock in 2014</h2>
        <iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
src="''' + plot1 + '''.embed?width=800&height=550"></iframe>
        <p>Apple stock price rose steadily through 2014.</p>
        
    </body>
</html>'''
# mp_print('Hi','i started here :)')
# mp_print({'table':\
#           (['Number','Multiplication_number','Result'],[range(1,11),range(2,22,2),np.arange(1,11)*np.arange(2,22,2)])})
# mp_print('bye','i am done')

# plot_url = first_plot()


# 
# import plotly.io as pio

# fig = go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1]))

# 

# 
# # html_str = insert_html(plot_url)


# # pio.write_html(fig, file='hello_world.html', auto_open=True)
