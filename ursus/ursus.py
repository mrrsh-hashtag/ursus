import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from prophet import Prophet
from scipy import interpolate
from sklearn.linear_model import LinearRegression, Ridge
# from ursus.ridge import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

class Ursus:
    def __init__(self, data, parameters):
        self.raw_data = self.load_data(data)
        self.parameters = self.validate_parameters(parameters)
        self.parameters["use_prophet"] = self.parameters.get("use_prophet", True)
        self.epsilon = 0.001
        self.gamma_ts = None

    def load_data(self, data):
        if isinstance(data, str):
            # If data is a string, assume it's a file path.
            if data.endswith(".csv"):
                return pd.read_csv(data)
            elif data.endswith(".xlsx"):
                return pd.read_excel(data)
        elif isinstance(data, pd.DataFrame):
            return data
        raise AttributeError("Data must be a data path or data frame")
    
    def validate_parameters(self, parameters):
        parameters["budget"] = parameters.get("budget", 5000)
        return parameters
    
    def train(self, plot=True):
        self.splited_data = self.split()
        self.rmse_list = []
        self.rssd_list = []
        self.r2_list = []
        self.rmse_rssd_list = []

        search_space = {}
        for ind_vad in self.parameters["ind_var"]:
            search_space[ind_vad + "_alpha"] = ng.p.Scalar(lower=1.0, upper=5)
            search_space[ind_vad + "_gamma"] = ng.p.Scalar(lower=0.0, upper=0.99)
            search_space[ind_vad + "_theta"] = ng.p.Scalar(lower=0.0, upper=0.99)
        search_space["ridge_alpha"] = ng.p.Scalar(lower=0.0, upper=1.99)

        parametrization = ng.p.Instrumentation(
            **search_space
        )

        optimizer = ng.optimizers.NGOpt(parametrization=parametrization,
                                        budget=self.parameters["budget"],
                                        num_workers=1)
        self.start_progress()
        recommendation = optimizer.minimize(self.model_candidate_train_test)
        print()
        
        self.hyper_params = recommendation.value[1]
        self.clf, r2, loss = self.model_candidate_train_test(return_model=True,
                                                             progress=False,
                                                             **self.hyper_params)
        print(f"Training r2: {r2:0.4} \t loss: {loss:0.4}")
        if plot:
            # self.get_response_curve()
            self.plot_one_page()
            # self.get_history_decomposition()
    
    def model_candidate_train_test(self, return_model=False, progress=True, **hyper_params):
        if progress:
            self.progress_tick()
        nr_spends = len(self.splited_data["ind_columns"])
        param_data = self.apply_params(hyper_params)
        ridge_alpha = hyper_params.get("ridge_alpha", 0.7)
                
        alpha = ridge_alpha
        # This is a good idea but not implemented in sklearns Ridge
        # alpha = np.zeros(param_data["train"]["x"].shape[1])
        # alpha[:nr_spends] = ridge_alpha
        clf = Ridge(alpha=alpha, fit_intercept=True)
        clf.fit(param_data["train"]["x"], param_data["train"]["y_hat"])
        y_pred = self.predict(param_data["test"]["x"], clf=clf, data="test")
        r2 = r2_score(param_data["test"]["y"], y_pred)
        if return_model:
            loss = self.loss_function(clf, y_pred, param_data, nr_spends)
            return clf, r2, loss
        else:
            self.r2_list.append(r2)
            loss = self.loss_function(clf, y_pred, param_data, nr_spends, save_values=True)
            return loss

    def predict(self, X, clf=None, data="train"):
        clf = clf if clf is not None else self.clf
        y_pred = clf.predict(X)
        if self.parameters["use_prophet"]:
            y_pred += self.splited_data[data]["y_season"]
        return y_pred

    def loss_function(self, clf, y_pred, param_data, nr_spends, save_values=False):
        # DECOMP.RSSD is short for "decomposition root sum of squared 
        # distance", a metric we "invented" to account for business logic.
        # The intuition is this: assuming you're spending 90% on TV and 
        # 10% on FB. If you get 10% effect for TV and 90% for FB, you'd 
        # probably not believe this result, no matter how low the model
        # error (NRMSE) is. If you get 80% TV and 20% FB as effect share,
        # it'll more "realistic". This is where the logic is from: 
        # minimising the distance between share of spend and share of 
        # effect.
        if "test" in param_data:
            y = param_data["test"]["y"]
            x = param_data["test"]["x"]
        else:
            y = param_data["y"]
            x = param_data["x"]
        rmse = np.linalg.norm(y_pred - y) / np.sqrt(len(y))
        rmse /= np.max(y) - np.min(y)
        # root sum of squared distance
        spend_share = np.sum(x[:, :nr_spends], axis=0)
        spend_share /= np.sum(spend_share)
        effect_share = clf.coef_[:nr_spends]
        effect_share /= np.sum(clf.coef_)
        rssd = np.sqrt(np.sum((effect_share - spend_share)**2))
        
        if save_values:
            self.rmse_list.append(rmse)
            self.rssd_list.append(rssd)
            self.rmse_rssd_list.append(np.sqrt(rssd**2 + rmse**2))

        ## Nevergrad algorithm selection
        # We've conducted a high-level comparison of Nevergrad algorithms
        # aiming to identify the best option for Robyn. At first, we ran
        # 500 iterations and 10 trials for all options. X axis is 
        # accumulated seconds and Y axis is combined loss function 
        # sqrt(sum(RMSE^2,DECOMP.RSSD^2)). The lower the combined error,
        # the better the model. We can observe that DE (Differential 
        # kcEvolution) and TwoPointsDE are not only achieving the lowest
        # error. They also show the ability to improve continuously, as
        # opposed to OnePlusOne or cGA, for example, that are reaching
        # convergence early and stop evolving. 
        objective = np.sqrt(rssd**2 + rmse**2)
        return objective

    def apply_params(self, hyper_params, use_split=True, x_data=None):
        x_test = None
        set_gamma = False
        if use_split:
            x_train = self.splited_data["train"]["x"]
            x_test = self.splited_data["test"]["x"]
            set_gamma = True
        elif x_data is not None:
            x_train = x_data.copy()
        else:
            x_train = self.splited_data["all"]["x"]
        
        x_train = self.apply_params_data(x_train, hyper_params, set_gamma_t=set_gamma)
        x_test = self.apply_params_data(x_test, hyper_params)
        
        if use_split:
            if not self.parameters["use_prophet"]:
                # Apply the week / season data
                x_train = np.concatenate([x_train, self.splited_data["train"]["week_mat"]], axis=1)
                x_test = np.concatenate([x_test, self.splited_data["test"]["week_mat"]], axis=1)
            return {
                "train": {
                    "x": x_train,
                    "y": self.splited_data["train"]["y"],
                    "y_season": self.splited_data["train"]["y_season"],
                    "y_hat": self.splited_data["train"]["y_hat"],
                },
                "test": {
                    "x": x_test,
                    "y": self.splited_data["test"]["y"],
                    "y_season": self.splited_data["test"]["y_season"],
                    "y_hat": self.splited_data["test"]["y_hat"],
                }
            }
        else:
            if not self.parameters["use_prophet"]:
                x_train = np.concatenate([x_train, self.splited_data["all"]["week_mat"]], axis=1)
            return {
                    "x": x_train,
                    "y": self.splited_data["all"]["y"],
                    "y_season": self.splited_data["all"]["y_season"],
                    "y_hat": self.splited_data["all"]["y_hat"],
                    "profet": self.splited_data["all"]["profet"]
                }
    
    def apply_params_data(self, x, hyper_params, set_gamma_t=False):
        if x is None:
            return
        done_with = []
        for hyper_param in hyper_params.keys():
            if hyper_param == "ridge_alpha":
                continue
            head = "_".join(hyper_param.split("_")[0:-1])
            if not head in done_with:
                head_idx = self.splited_data["ind_columns"][head]
                alpha = hyper_params[head + "_alpha"]
                gamma = hyper_params[head + "_gamma"]
                theta = hyper_params[head + "_theta"]
                x = self.apply_adstock(head_idx, theta, x)
                x = self.apply_saturation(head_idx, alpha, gamma, x, set_gamma_t)
                done_with.append(head)
        return x

    def apply_adstock(self, idx, theta, x):
        ad_stock_value = 0
        x = x.copy()
        for i in range(x.shape[0]):
            ad_stock_value = x[i, idx] + ad_stock_value * theta
            x[i, idx] = ad_stock_value
        return x

    def apply_saturation(self, idx, alpha, gamma, x, set_gamma_t):
        x_i = x[:, idx]
        
        if set_gamma_t:
            max_i = np.max(x_i)
            min_i = np.min(x_i)
            gammaTrans = np.quantile(np.linspace(min_i, max_i, 10000), gamma)
            if self.gamma_ts is None:
                self.gamma_ts = np.zeros((x.shape[1],))
            self.gamma_ts[idx] = gammaTrans
        else:
            gammaTrans = self.gamma_ts[idx]
                            
        x_j = (x_i**alpha)/((gammaTrans**alpha) + x_i**alpha + self.epsilon)
        x[:, idx] = x_j
        return x

    def split(self):
        train_df, test_df = train_test_split(self.raw_data, test_size=0.2, shuffle=False)
        
        # Fit the prophet model to the training data and use it to remove 
        # seasonality effects. The rest should be explained using Ridge 
        # regression.
        date_var = self.parameters["date_var"]
        dep_var = self.parameters["dep_var"]
        pro_df_trn = train_df[[date_var, dep_var]]
        pro_df_trn = pro_df_trn.rename(columns={date_var: "ds", dep_var: "y"})
        pro_df_tst = test_df[[date_var, dep_var]]
        pro_df_tst = pro_df_tst.rename(columns={date_var: "ds", dep_var: "y"})
        pro_df_all = self.raw_data[[date_var, dep_var]]
        pro_df_all = pro_df_all.rename(columns={date_var: "ds", dep_var: "y"})

        if self.parameters["use_prophet"]:
            log.info("Using prophet")
            country_code = self.parameters.get("country_code", False)
            if country_code:
                log.info(f"Using holidays for {country_code}")
                holidays = pd.read_csv("prophet_holidays.csv")
                prophet_model = Prophet(holidays=holidays,
                                        # weekly_seasonality=True,
                                        # daily_seasonality=False
                                        )
            else:
                log.info("Not using holidays")
                prophet_model = Prophet(weekly_seasonality=True,
                                        daily_seasonality=False)
                
            prophet_model.fit(pro_df_trn)
            pro_df_trn_pred = prophet_model.predict(pro_df_trn[["ds"]])
            pro_df_tst_pred = prophet_model.predict(pro_df_tst[["ds"]])
            pro_df_all_pred = prophet_model.predict(pro_df_all[["ds"]])
        
            y_season_trn = pro_df_trn_pred["yearly"].values
            y_season_tst = pro_df_tst_pred["yearly"].values
            y_season_all = pro_df_all_pred["yearly"].values

            y_hat_trn = train_df[self.parameters["dep_var"]].values - y_season_trn
            y_hat_tst = test_df[self.parameters["dep_var"]].values - y_season_tst
            y_hat_all = self.raw_data[self.parameters["dep_var"]].values - y_season_all

            # df = pro_df_trn_pred
            # ax = plt.gca()
            # ax.plot(df["ds"], pro_df_trn["y"], label="y")
            # ax.plot(df["ds"], df["yearly"], label="year")
            # ax.plot(df["ds"], y_hat_trn, label="y-year")
            # ax.legend()
            # plt.show()
            
            week_mat_trn = 0
            week_mat_tst = 0
            week_mat_all = 0

        else:
            (pro_df_all_pred, y_season_trn, y_season_tst, y_season_all) = [None] * 4
            y_hat_trn = train_df[self.parameters["dep_var"]].values
            y_hat_tst = test_df[self.parameters["dep_var"]].values
            y_hat_all = self.raw_data[self.parameters["dep_var"]].values
            # Calculating week number matrices
            week_mat_trn = self.get_week_mat(pro_df_trn["ds"])
            week_mat_tst = self.get_week_mat(pro_df_tst["ds"])
            week_mat_all = self.get_week_mat(pro_df_all["ds"])
        

        return {
            "train": {
                "x": train_df[self.parameters["ind_var"]].values,
                "y": train_df[self.parameters["dep_var"]].values,
                "y_season": y_season_trn,
                "y_hat": y_hat_trn,
                "week_mat": week_mat_trn
            },
            "test": {
                "x": test_df[self.parameters["ind_var"]].values,
                "y": test_df[self.parameters["dep_var"]].values,
                "y_season": y_season_tst,
                "y_hat": y_hat_tst,
                "week_mat": week_mat_tst
            },
            "all": {
                "x": self.raw_data[self.parameters["ind_var"]].values,
                "y": self.raw_data[self.parameters["dep_var"]].values,
                "y_season": y_season_all,
                "y_hat": y_hat_all,
                "profet": pro_df_all_pred,
                "week_mat": week_mat_all
            },
            "ind_columns": {
                col: self.parameters["ind_var"].index(col) for col in self.parameters["ind_var"]
            },
            "gammaTrans": {}
        }
    
    def get_week_mat(self, ds):
        dts = pd.to_datetime(ds, infer_datetime_format=True)
        week_mat = np.zeros((dts.size, 53))
        for i, week in enumerate(dts.dt.isocalendar().week):
            week_mat[i, week - 1] = 1
        # Delete the first week and use it as a reference point
        week_mat = np.delete(week_mat, 0, 1)
        return week_mat

    def plot_one_page(self):
        fig, axs = plt.subplots(figsize=(12, 6), nrows=2, ncols=2)
        axs = axs.reshape(-1)

        self.plot_regression(axs[0])
        self.plot_pareto_front(axs[1])
        self.plot_training_eval(axs[2])
        self.plot_spend_effect(axs[3])
        plt.tight_layout()
        plt.savefig("Training and testing plot",
                facecolor="white",
                dpi=400,
                )
        plt.show()
    
    def plot_pareto_front(self, ax):
        color = np.linspace(0, 1, len(self.rmse_list))
        cmap = plt.get_cmap("turbo", len(self.rmse_list))
        ax.scatter(self.rmse_list, self.rssd_list, alpha=0.3, c=color, cmap=cmap)
        
        
        param_data = self.apply_params(self.hyper_params, use_split=False)
        y_pred = self.predict(param_data["x"], data="all")
        nr_spends = len(self.splited_data["ind_columns"])
        loss = self.loss_function(self.clf, y_pred, param_data, nr_spends)

        ax.set_title(f"Pareto front. Final loss: {loss:0.4}")
        ax.set_xlabel("rmse")
        ax.set_ylabel("rssd")
        
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, len(self.rmse_list), 10),
                            boundaries=np.arange(-0.05,2.1,.1),
                            orientation="vertical",
                            ax=ax)

    def plot_training_eval(self, ax):
        ax.plot(self.r2_list, label=r"$r^2$")
        ax.plot(self.rmse_rssd_list, label=r"$\sqrt{rmse^2 + rssd^2}$")
        ax.plot(self.rmse_list, label="NRMSE")
        ax.plot(self.rssd_list, label="RSSD")
        
        ax.legend()
        ax.set_title(r"Training evaluation. Final $r^2=$" + f"{self.r2_list[-1]:0.4}")

    def plot_spend_effect(self, ax):
        labels = list(self.splited_data["ind_columns"].keys())
        x = np.arange(len(labels))

        param_data = self.apply_params(self.hyper_params)
        spend_share = np.sum(param_data["test"]["x"][:, :len(labels)], axis=0)
        spend_share /= np.sum(spend_share)
        print(spend_share)
        spend_share = np.sum(self.splited_data["test"]["x"][:, :len(labels)], axis=0)
        spend_share /= np.sum(spend_share)
        effect_share = self.clf.coef_[:len(labels)]
        effect_share /= np.sum(effect_share)
        
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width/2, spend_share, width, label="Spend share")
        rects2 = ax.bar(x + width/2, effect_share, width, label="Effect share")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title("Spend vs. effect")
        ax.set_xticks(x, labels)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy() 
            ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')
    
    def plot_regression(self, ax):
        param_data = self.apply_params(self.hyper_params)
        y_pred_trn = self.predict(param_data["train"]["x"], data="train")
        y_pred_tst = self.predict(param_data["test"]["x"], data="test")

        x_tst = range(0, y_pred_tst.size)
        x_trn = range(y_pred_tst.size, y_pred_trn.size + y_pred_tst.size)
        x_all = range(0, y_pred_trn.size + y_pred_tst.size)

        y_true_trn = param_data["train"]["y"]
        y_true_tst = param_data["test"]["y"]
        
        y_true = np.concatenate((y_true_tst, y_true_trn))

        r2 = r2_score(y_true, np.concatenate((y_pred_tst, y_pred_trn)))

        ax.plot(x_trn, y_pred_trn, label="y-pred (training)")
        ax.plot(x_all, y_true, label="y true")
        ax.plot(x_tst, y_pred_tst, label="y-pred (testing)")
        if self.parameters["use_prophet"]:
            ax.plot(x_tst, self.splited_data["test"]["y_season"], "--")
            ax.plot(x_trn, self.splited_data["train"]["y_season"], "--")
            ax.plot(x_all, self.splited_data["all"]["y_hat"], label=r" $\hat{y}$")
            

        ax.legend()
        ax.set_title(r"Line following. Total $r^2=$" + f"{r2:0.4}")
        
    def get_training_data(self):
        training_data = {
            "spend_vs_effect": self.get_spend_vs_effect(),
            "history_decomposition": self.get_history_decomposition(),
            "hyper_params": self.hyper_params,
            "response_curve": self.get_response_curve(),
            "metrics": self.get_training_metrics()
        }
        return training_data

    def get_spend_vs_effect(self):
        labels = list(self.splited_data["ind_columns"].keys())
        spend_share = np.sum(self.splited_data["all"]["x"][:, :len(labels)], axis=0)
        spend_share /= np.sum(spend_share)
        effect_share = self.clf.coef_
        effect_share /= np.sum(effect_share)
        data = []
        for i, label in enumerate(labels):
            data.append(
                {
                    "name": label,
                    "effect_share": effect_share[i],
                    "spend_share": spend_share[i]
                }
            )
        return data

    def get_history_decomposition(self):
        date_var = self.parameters["date_var"]
        dep_var = self.parameters["dep_var"]
        spend_labels = list(self.splited_data["ind_columns"].keys())
        nr_spends = len(spend_labels)
                
        param_data = self.apply_params(self.hyper_params, use_split=False)

        values = param_data["x"] * self.clf.coef_
        spends = values[:, :nr_spends]
        if not self.parameters["use_prophet"]:
            season = np.sum(values[:, nr_spends:], axis=1)
        else:
            season = self.splited_data["all"]["y_season"]
        
        intercept = self.clf.intercept_ * np.ones_like(season)

        # Scale the spend values to fit actual y
        x = (param_data["y"] - intercept - season) / np.sum(spends, axis=1)
        spends = (spends.T * x).T
        values = np.concatenate([spends,
                                season.reshape(-1, 1),
                                intercept.reshape(-1, 1)], axis=1)

        values = pd.DataFrame(values, columns = spend_labels + ["Season", "Intercept"])
        values["Date"] = self.raw_data[date_var]
        values["y"] = self.raw_data[dep_var]
        
        y_span = [0, param_data["y"].max() * 1.1]
        # Reorder columns
        col_mean = sorted([(values[lbl].mean(), lbl) for lbl in spend_labels], reverse=True)
        col_order = ["Date", "Intercept", "Season"] + [col[1] for col in col_mean] + ["y"]
        values = values[col_order]
        labels = list(values.columns.values)
        values = values.to_dict("records")
        
        hist_decomp = {
            "dataKey": "Date",
            "oyLabel": dep_var,
            "oxLabel": "Date",
            "yLimit": y_span,
            "values": values,
            "labels": labels
        }
        return hist_decomp



        # for i, date in self.raw_data["ds"].values:
        #     item = {
        #         "date": date,

        #     }

    def get_response_curve(self):
        x_spend = np.zeros((100, len(self.splited_data["ind_columns"])))
        for i, col in enumerate(self.splited_data["ind_columns"]):
            x_spend[:, i] = np.linspace(0, np.max(self.raw_data[col]), 100)
        
        x_response = self.apply_params(self.hyper_params, use_split=False, x_data=x_spend)["x"]

        curves = {}
        for i, col in enumerate(self.splited_data["ind_columns"]):
            curves[col] = {
                "mean": np.mean(x_spend[:, i]),
                "curve": [{"s": x_spend[j, i], "r": x_response[j, i]} for j, _ in enumerate(x_spend[:, i])]
            }
        
        return curves

    def get_training_metrics(self):
        return {
            "r2": self.r2_list[-1],
            "rmse_rssd": self.rmse_rssd_list[-1]
        }

    def get_budget_optimum(self):
        # X_opt = lambda * (alpha - 1)^(1/alpha)
        pass


    def progress_tick(self):
        self.count += 1
        # This breaks in the cloud.
        #size = os.get_terminal_size().columns
        #nr_ticks = int(size * self.count / self.parameters["budget"])
        #sys.stdout.write("\033[F") #back to previous line 
        #sys.stdout.write("\033[K")
        #print("#"*nr_ticks)

    def start_progress(self):
        self.count=0

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """ add an arrow to a line.
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
