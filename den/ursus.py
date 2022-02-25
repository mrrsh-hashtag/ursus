import copy
import logging
import os
import sys

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from den.ridge import HNRidge

log = logging.getLogger(__name__)

class Ursus:
    EPS = 0.001     # Epsilon small value for numerical stability
    DEFAULT_PARAMS = {
        "budget": 5000,
        "adstock_algo": "geometric",
        "use_prophet": True,
        "print_progress": False,
        "plot_one_page": False,
        "auto_weight_season": False,
        "trials": 1,
        "thetas": {}
    }

    def __init__(self, data, parameters):
        self.raw_data = self.load_data(data)
        self.parameters = self.validate_parameters(parameters)
        self.gamma_ts = None
        self.budget_optimizations = []

    def load_data(self, data):
        if isinstance(data, str):
            # If data is a string, assume it's a file path.
            if data.endswith(".csv"):
                return self._cast_data(pd.read_csv(data))
            elif data.endswith(".xlsx"):
                return self._cast_data(pd.read_excel(data))
        elif isinstance(data, pd.DataFrame):
            return data
        raise AttributeError("Data must be a data path or data frame")
    
    def _cast_data(self, df):
        cols = df.columns
        for col in cols:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
        return df

    def validate_parameters(self, parameters):
        params = self.DEFAULT_PARAMS
        params.update(parameters)
        mandatory_keys = ["dep_var", "date_var", "ind_var"]
        for mand_key in mandatory_keys:
            if mand_key not in params.keys():
                msg = f"Key '{mand_key}' is mandatory but not found in params"
                log.error(msg)
                raise AttributeError(msg)
        return params

    def workshop_test(self):
        self.splited_data = self.split()
        hyper_params = {
            "tv_S_alpha" : 1.2430779259509903,
            "tv_S_gamma" : 0.2499373542652512,
            "tv_S_theta" : 0.3876019745820854,
            "ooh_S_alpha" : 4.916248265121615,
            "ooh_S_gamma" : 0.3081468802299278,
            "ooh_S_theta" : 0.7781770519983228,
            "print_S_alpha" : 2.4718420113105664,
            "print_S_gamma" : 0.4742820947613169,
            "print_S_theta" : 0.5173008891158258,
            "search_S_alpha" : 1.929375063592007,
            "search_S_gamma" : 0.1723375354690649,
            "search_S_theta" : 0.1683844649758252,
            "facebook_S_alpha" : 4.669184879925492,
            "facebook_S_gamma" : 0.05481381725765699,
            "facebook_S_theta" : 0.06777067628975211
        }
        param_data = self.apply_params(hyper_params)
        
        # clf = HNRidge(fit_intercept=False, scale=True)
        # clf.fit(param_data["train"].get_indeps(), param_data["train"].y_hat)
        theta = 0.3876019745820854
        sigma = 8   # Slope
        tau = 5 # Half time
        g = self._geometric(theta, 100)
        w = self._weibull(sigma, tau, 100)

        x = np.zeros(100)
        xw = np.zeros(100)
        xg = np.zeros(100)
        x[50:60] = 10

        for i in range(x.size):
            # g = self._geometric(theta, x.size - i)
            print(g[:x.size - i])
            xg[i] = np.sum(x[i:] * g[:x.size - i])
            xw[i] = np.sum(x[i:] * w[:x.size - i])
        plt.plot(xg)
        plt.plot(xw)
        plt.show()

    def train(self):
        log.info(f"Training start. Budget: {self.parameters['budget']} Trials: {self.parameters['trials']}")
        self.splited_data = self.split()
        self.metrics = {
            "rmse": [],
            "rssd": [],
            "r2": [],
            "rmse_rssd": []
        }
        
        hyper_params = None
        for trial in range(self.parameters["trials"]):
            search_params = self.get_search_params(hyper_params)
            optimizer = ng.optimizers.registry["TwoPointsDE"](
                parametrization=search_params,
                budget=self.parameters["budget"],
                num_workers=1
            )
            # optimizer = ng.optimizers.NGOpt()
            self.start_progress()
            recommendation = optimizer.minimize(self.model_candidate_train_test)
            hyper_params = recommendation.value[1]
            self.clf, r2, self.loss = self.model_candidate_train_test(return_model=True,
                                                                progress=False,
                                                                **hyper_params)
            log.info(f"Trial {trial} end. r2: {r2:0.4f}, loss: {self.loss:0.4f}")
        self.hyper_params = hyper_params

        if self.parameters["plot_one_page"]:
            self.plot_one_page()
    
    def get_search_params(self, hyper_params, noise=0.1):
        search_space = {}
        if not hyper_params is None:
            for key, value in hyper_params.items():
                lower, upper = self._limit_bounds(key, value, noise)
                search_space[key] = ng.p.Scalar(lower=lower, upper=upper)
        else:
            for ind_var in self.parameters["ind_var"]:
                search_space[ind_var + "_alpha"] = ng.p.Scalar(lower=1.0, upper=5)
                search_space[ind_var + "_gamma"] = ng.p.Scalar(lower=0.0, upper=0.99)
                if self.parameters["adstock_algo"] == "weibull":
                    t_lower = self.parameters["taus"].get(ind_var, {}).get("lower", 0.5)
                    t_upper = self.parameters["taus"].get(ind_var, {}).get("upper", 8.0)
                    search_space[ind_var + "_sigma"] = ng.p.Scalar(lower=0.5, upper=8.0)
                    search_space[ind_var + "_tau"] = ng.p.Scalar(lower=t_lower, upper=t_upper)
                else:
                    lower = self.parameters["thetas"].get(ind_var, {}).get("lower", 0)
                    upper = self.parameters["thetas"].get(ind_var, {}).get("upper", 0.99)
                    search_space[ind_var + "_theta"] = ng.p.Scalar(lower=lower, upper=upper)
        # search_space["ridge_alpha"] = ng.p.Scalar(lower=0.0, upper=1.99)

        parametrization = ng.p.Instrumentation(
            **search_space
        )
        return parametrization

    def _limit_bounds(self, key, value, noise):
        lower = value * (1 - noise)
        upper = value * (1 + noise)
        if "_gamma" in key or "_theta" in key:
            lower = lower if lower > 0 else 0
            upper = upper if upper < 1 else 1
        return lower, upper

    def model_candidate_train_test(self, return_model=False, progress=True, **hyper_params):
        if progress:
            self.progress_tick()
        nr_spends = len(self.splited_data["ind_columns"])
        param_data = self.apply_params(hyper_params)
        ridge_alpha = hyper_params.get("ridge_alpha", 0.7)
                
        alpha = ridge_alpha
        clf = HNRidge(alpha=alpha, fit_intercept=False, scale=True)
        clf.fit(param_data["train"].get_indeps(), param_data["train"].y_hat)
        y_pred = self._predict(param_data["train"], clf=clf)
        r2 = r2_score(param_data["train"].y, y_pred)
        
        if return_model:
            loss = self.loss_function(clf, y_pred, param_data, nr_spends)
            return clf, r2, loss
        else:
            self.metrics["r2"].append(r2)
            loss = self.loss_function(clf, y_pred, param_data, nr_spends, save_metrics=True)
            return loss
    
    def _predict(self, dataset, clf=None):
        clf = clf if clf is not None else self.clf
        X = dataset.get_indeps()
        y_pred = clf.predict(X) + dataset.y_season
        return y_pred

    def loss_function(self, clf, y_pred, param_data, nr_spends, save_metrics=False):
        # DECOMP.RSSD is short for "decomposition root sum of squared 
        # distance", a metric we "invented" to account for business logic.
        # The intuition is this: assuming you're spending 90% on TV and 
        # 10% on FB. If you get 10% effect for TV and 90% for FB, you'd 
        # probably not believe this result, no matter how low the model
        # error (NRMSE) is. If you get 80% TV and 20% FB as effect share,
        # it'll more "realistic". This is where the logic is from: 
        # minimising the distance between share of spend and share of 
        # effect.

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

        # Special condition where spend betas won't be allowed to be negative.
        # Don't save metrics if thats the case
        # if np.any(clf.coef_[:nr_spends] < 0):
        #     print(clf.coef_[:nr_spends])
        #     print(clf.coef_[:nr_spends] < 0)
        #     print()
        #     return 1000

        if isinstance(param_data, UrsusDataset):
            y = param_data.y
            x = param_data.x
            X = param_data.get_indeps()
            y_hat = param_data.y_hat
        else:
            y = param_data["train"].y
            x = param_data["train"].x
            X = param_data["train"].get_indeps()
            y_hat = param_data["train"].y_hat
        y_pred = clf.predict(X)
        # Measure the "inner" mse. Without season. Seems to work better.
        rmse = np.linalg.norm(y_pred - y_hat) / np.sqrt(len(y_hat))
        rmse /= np.max(y_hat) - np.min(y_hat)
        # rmse = np.linalg.norm(y_pred - y) / np.sqrt(len(y))
        # rmse /= np.max(y) - np.min(y)

        # root sum of squared distance
        spend_share = np.sum(x[:, :nr_spends], axis=0)
        spend_share /= np.sum(spend_share)
        effect_share = clf.coef_[:nr_spends]
        effect_share /= np.sum(clf.coef_)
        rssd = np.sqrt(np.sum((effect_share - spend_share)**2))

        # Penalize negatvie spend values:
        neg_factor = np.sum(effect_share < -0.0) / nr_spends

        objective = np.sqrt(rssd**2 + rmse**2 + neg_factor**2)

        if save_metrics:
            self.metrics["rmse"].append(rmse)
            self.metrics["rssd"].append(rssd)
            self.metrics["rmse_rssd"].append(objective)
                    
        return objective

    def apply_params(self, hyper_params, use_split=True, x_data=None):
        x_test = None
        set_gamma = False
        if use_split:
            x_train = self.splited_data["train"].x
            x_test = self.splited_data["test"].x
            set_gamma = True
        elif x_data is not None:
            x_train = x_data.copy()
        else:
            x_train = self.splited_data["all"].x
        
        x_train = self.apply_params_data(x_train, hyper_params, set_gamma_t=set_gamma)
        x_test = self.apply_params_data(x_test, hyper_params)
        
        if use_split:
            train_ds = copy.deepcopy(self.splited_data["train"])
            train_ds.x = x_train
            test_ds = copy.deepcopy(self.splited_data["test"])
            test_ds.x = x_test
            
            return {
                "train": train_ds,
                "test": test_ds
            }
        else:
            all_ds = copy.deepcopy(self.splited_data["all"])
            all_ds.x = x_train
            return all_ds
    
    def apply_params_data(self, x, hyper_params, set_gamma_t=False,
                          apply_adstock=True, apply_saturation=True,
                          multiply_sat=True):
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
                theta, sigma, tau = None, None, None
                if self.parameters["adstock_algo"] == "weibull":
                    sigma = hyper_params[head + "_sigma"]
                    tau = hyper_params[head + "_tau"]
                else:
                    theta = hyper_params[head + "_theta"]
                if apply_adstock:
                    x = self.apply_adstock(head_idx, x, theta, sigma, tau)
                if apply_saturation:
                    x = self.apply_saturation(head_idx, alpha, gamma, x,
                                              set_gamma_t, multiply_sat)
                done_with.append(head)
        return x

    def apply_adstock(self, idx, x, theta=None, sigma=None, tau=None):
        """
        https://towardsdatascience.com/carryover-and-shape-effects-in-media-mix-modeling-paper-review-fd699b509e2d#2449
        """
        x = x.copy()
        if self.parameters["adstock_algo"] == "weibull":
            weights = self._weibull(sigma, tau, x[:, idx].size)
        else:
            weights = self._geometric(theta, x[:, idx].size)
        x_ads = np.zeros_like(x[:, idx])
        for i in range(x_ads.size):
            x_ads[i] = np.sum(x[i:, idx] * weights[:x_ads.size - i])
        x[:, idx] = x_ads
        return x
    
    def _weibull(self, sigma, tau, size, resolution=1):
        """ Tau: halftime
            Sigma: slope
        """
        t = np.arange(0, size, resolution)
        lam = tau / np.power(0.5, 1 / sigma)
        return np.power(np.e, -np.power(t/lam, sigma))
    
    def _geometric(self, theta, size, resolution=1):
        t = np.arange(0, size, resolution)
        return np.power(theta, t)

    def apply_saturation(self, idx, alpha, gamma, x, set_gamma_t, multiply_sat):
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
                            
        x_j = (x_i**alpha)/((gammaTrans**alpha) + x_i**alpha + self.EPS)
        if multiply_sat:
            x[:, idx] *= x_j
        else:
            x[:, idx] = x_j

        return x

    def split(self):
        train_df, test_df = train_test_split(
            self.raw_data,
            test_size=0.2,
            shuffle=False
        )
        
        # Fit the prophet model to the training data and use it to remove 
        # seasonality effects. The rest should be explained using Ridge 
        # regression.
        train_ds = UrsusDataset(train_df, self.parameters)
        test_ds = UrsusDataset(test_df, self.parameters)
        all_ds = UrsusDataset(self.raw_data, self.parameters)
        
        prophet_model = train_ds.get_prophet()
        train_ds.set_prophet(prophet_model)
        test_ds.set_prophet(prophet_model)
        all_ds.set_prophet(prophet_model)

        ind_columns = {}
        for col in self.parameters["ind_var"]:
            ind_columns[col] = self.parameters["ind_var"].index(col)

        return {
            "train": train_ds,
            "test": test_ds,
            "all": all_ds,
            "ind_columns": ind_columns,
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
    
    def get_pareto_data(self):
        rssd = np.array(self.metrics["rssd"])
        rmse = np.array(self.metrics["rmse"])
        c_vals = np.arange(rssd.size)
        return {
            "rssd": rssd,
            "rmse": rmse,
            "c_vals": c_vals,
            "loss": self.loss
        }

    def get_training_eval_data(self):
        x = list(range(len(self.metrics["r2"])))
        return {
            "x": x,
            "r2": self.metrics["r2"],
            "rmse_rssd": self.metrics["rmse_rssd"],
            "rmse": self.metrics["rmse"],
            "rssd": self.metrics["rssd"]
        }
    
    def get_spend_effect_data(self):
        labels = list(self.splited_data["ind_columns"].keys())
        spend_share = np.sum(self.splited_data["all"].x[:, :len(labels)], axis=0)
        spend_share /= np.sum(spend_share)
        effect_share = self.clf.coef_[:len(labels)]
        effect_share /= np.sum(effect_share)
        return {
            "spend_share": spend_share,
            "effect_share": effect_share,
            "labels": labels
        }

    def get_regression_data(self):
        param_data = self.apply_params(self.hyper_params, use_split=False)
        y_pred = self._predict(param_data)
        r2 = r2_score(param_data.y, y_pred)
        return {
            "y_pred": y_pred,
            "y_true": param_data.y,
            "x_date": param_data.dates,
            "y_season": param_data.y_season,
            "r2": r2
        }

    def get_waterfall_data(self):
        param_data = self.apply_params(self.hyper_params,
                                       use_split=False)
        xc_values = param_data.get_indeps()
        xc_mean = np.mean(xc_values * self.clf.coef_, axis=0).tolist()
        xc_cols = param_data.get_indeps_cols()

        xc_mean.append(self.clf.intercept_)
        xc_cols.append("Intercept")
                
        xc_mean.append(np.mean(np.abs(param_data.y_season)))
        xc_cols.append("Season")

        xc_mean, xc_cols = zip(*sorted(zip(xc_mean, xc_cols), reverse=True))
        data = {"amount": xc_mean}
        wf_df = pd.DataFrame(data=data,index=xc_cols)
        return wf_df
    
    def get_adstock_data(self):
        size = 10
        res = 0.2
        done_with = {
            "time_lag": np.arange(0, size, res).tolist()
        }
        for hyper_param in self.hyper_params.keys():
            if hyper_param == "ridge_alpha":
                continue
            head = "_".join(hyper_param.split("_")[0:-1])
            if not head in done_with:
                head_idx = self.splited_data["ind_columns"][head]
                if self.parameters["adstock_algo"] == "weibull":
                    sigma = self.hyper_params[head + "_sigma"]
                    tau = self.hyper_params[head + "_tau"]
                    weights = self._weibull(sigma, tau, size, resolution=res)
                else:
                    theta = self.hyper_params[head + "_theta"]
                    weights = self._geometric(theta, size, resolution=res)
                done_with[head] = weights.tolist()
        return done_with

    def get_training_data(self):
        waterfall_data = self.get_waterfall_data()
        waterfall_data = waterfall_data.to_dict()
        training_data = {
            "spend_vs_effect": self.get_spend_vs_effect(),
            "history_decomposition": self.get_history_decomposition(),
            "hyper_params": self.hyper_params,
            "response_curve": self.get_response_curve(),
            "metrics": self.get_training_metrics(),
            "budget_split_optimum": self.get_budget_optimization(),
            "budget_split_optimum_high_risk": self.get_budget_optimization(0.6, 1.4),
            "waterfall": waterfall_data,
            "adstock": self.get_adstock_data()
        }
        return training_data

    def get_spend_vs_effect(self):
        labels = list(self.splited_data["ind_columns"].keys())
        spend_share = np.sum(self.splited_data["all"].x[:, :len(labels)], axis=0)
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
        
        spends = self.apply_params_data(
            self.splited_data["all"].x,
            self.hyper_params,
            apply_adstock=True,
            apply_saturation=True)[:, :nr_spends]
                
        spends = spends * self.clf.coef_[:nr_spends]

        season = self.splited_data["all"].y_season.reshape(-1, 1)
        intercept = self.clf.intercept_ * np.ones_like(season)

        values = np.concatenate([spends,
                                season,
                                intercept], axis=1)

        values = pd.DataFrame(values, columns = spend_labels + ["Season", "Intercept"])
        values["Date"] = self.raw_data[date_var]
        values["y"] = self.raw_data[dep_var]
        
        y_span = [0, self.raw_data[dep_var].max() * 1.1]
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

    def get_response_curve(self):
        x_spend = np.zeros((100, len(self.splited_data["ind_columns"])))
        for i, col in enumerate(self.splited_data["ind_columns"]):
            x_spend[:, i] = np.linspace(0, np.max(self.raw_data[col]), num=100)
        
        x_response = self.apply_params_data(
            copy.copy(x_spend),
            self.hyper_params,
            apply_adstock=False,
            apply_saturation=True,
            multiply_sat=False
        )
        
        curves = {}
        for i, col in enumerate(self.splited_data["ind_columns"]):
            curves[col] = {
                "mean_s": np.mean(x_spend[:, i]),
                "mean_r": np.mean(x_response[:, i]),
                "curve": [{"s": x_spend[j, i], "r": x_response[j, i]} for j, _ in enumerate(x_spend[:, i])]
            }
        
        return curves

    def get_training_metrics(self):
        end_metrics = {}
        for met_key, met_val in self.metrics.items():
            end_metrics[met_key] = met_val[-1]
        return end_metrics

    def get_budget_optimization(self, l_bound=0.8, h_bound=1.2, budget=None, get_model=False):
        """
        Optimize budget spend and returning an array of floats. These values
        are to be interpreted as weights in the range [1-max_diff; 1+max_diff]
        that should be applied to the mean spend seen in the data.
        The max_diff variable is set to protect agains overreliance on the 
        model. Start positions of the weights are chosen randomly. There is 
        uncertainty on how much the random initiation impacts the results but 
        uniform weights of 1 have proven to find local minima.


        Should use https://facebookresearch.github.io/nevergrad/optimization.html#optimization-with-constraints
        and skip the omegas. CHECK
        """
        data = self.splited_data["all"]
        x_mean = np.mean(data.x, axis=0)
        budget = budget if budget is not None else np.sum(x_mean)
        for budget_optimization in self.budget_optimizations:
            l = budget_optimization["l_bound"]
            h = budget_optimization["h_bound"]
            b = budget_optimization["budget"]
            if l==l_bound and h==h_bound and b == budget:
                model = budget_optimization["model"]
                if get_model:
                    return model
                return {
                    "budgetChange": model.budget_change,
                    "effectChange": model.effect_change,
                    "allocation": model.allocation
                }        
        
        l_bounds = self._format_bounds(x_mean, l_bound)
        h_bounds = self._format_bounds(x_mean, h_bound)
        
        parametrization = ng.p.Instrumentation(
            xs=ng.p.Array(init=x_mean).set_bounds(lower=l_bounds, upper=h_bounds)
        )
        optimizer = ng.optimizers.registry["TwoPointsDE"](
            parametrization=parametrization,
            budget=2000
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            constraint = lambda x: np.sum(x[1]["xs"]) <= budget
            optimizer.parametrization.register_cheap_constraint(constraint)
        recommendation = optimizer.minimize(self._simp_func)
        
        opti = recommendation.value[1]["xs"]
        init = x_mean
        opti_effect = self._spend_effect(opti)
        init_effect = self._spend_effect(init)
        init_effect.merge(opti_effect)

        self.budget_optimizations.append(
            {
                "l_bound": 0.8,
                "h_bound": 1.2,
                "budget": budget,
                "model": init_effect
            }
        )
        if get_model:
            return init_effect
        ret_dict = {
            "budgetChange": init_effect.budget_change,
            "effectChange": init_effect.effect_change,
            "allocation": init_effect.allocation
        }
        return ret_dict
    
    def _format_bounds(self, init_val, bound):
        if isinstance(bound, (list, tuple)):
            bounds = np.array(bound)
        elif isinstance(bound, np.ndarray):
            bounds = bound
        elif isinstance(bound, (float, int)):
            n = self.splited_data["all"].x.shape[1]
            bounds = np.ones(n) * np.array(bound)
        else:
            msg = "bound needs to be either a scalar, list or array"
            log.error(msg)
            raise AttributeError(msg)
        return init_val * bounds

    def _simp_func(self, xs):
        effect_obj = self._spend_effect(xs)
        return 1 / effect_obj.total_effect
    
    def _spend_effect(self, xs):
        xs = list(xs)
        done_with = []
        spends = []
        for hyper_param in self.hyper_params.keys():
            if hyper_param == "ridge_alpha":
                continue
            head = "_".join(hyper_param.split("_")[0:-1])
            if not head in done_with:
                head_idx = self.splited_data["ind_columns"][head]
                alpha = self.hyper_params[head + "_alpha"]
                gamma = self.gamma_ts[head_idx]

                x_omega = xs[head_idx]
                                
                xs[head_idx] = x_omega * self.clf.coef_[head_idx] * \
                    np.power(x_omega, alpha) / (np.power(gamma, alpha) + \
                    np.power(x_omega, alpha))
                spends.append(x_omega)
                done_with.append(head)
        return EvalEffect(self.splited_data["ind_columns"], xs, spends, self.clf.intercept_)

    def progress_tick(self):
        if self.parameters["print_progress"]:
            self.count += 1
            size = os.get_terminal_size().columns
            nr_ticks = int(size * self.count / self.parameters["budget"])
            sys.stdout.write("\033[F") #back to previous line 
            sys.stdout.write("\033[K")
            print("#"*nr_ticks)

    def start_progress(self):
        if self.parameters["print_progress"]:
            print()
            self.count=0
    
    def end_progress(self):
        if self.parameters["print_progress"]:
            print()


class UrsusDataset:
    """
    The UrsusDataset is a helper class used to organize and clean the data to
    the Ursus regression. The data is divided up in
    x: The indipendent spent variables
    c: Context variables.
    y: The dependent variable, raw, as given from the input.
    y_season: The variations in the dependent variables that is to explained
        soley by the time of the year.
    y_hat: The dependent variable minus the seasonal variations. That is the
        dependet variable that is supposed to be explained by x and c.
    """
    def __init__(self, df, params):
        self.x = df[params["ind_var"]].values
        self.y = df[params["dep_var"]].values
        self.c = df[params["context"]].values
        self.dates = df[params["date_var"]].values
        self.params = params
        self.df = df
        
    def get_prophet(self):
        log.info("Using prophet")
        country_code = self.params.get("country_code", False)
        if country_code:
            log.info(f"Using holidays for {country_code}")
            log.warn("Holidays in prophet is untested")
            holidays = pd.read_csv("prophet_holidays.csv")
            prophet_model = Prophet(holidays=holidays,
                                    weekly_seasonality=True,
                                    daily_seasonality=False
                                    )
        else:
            log.info("Not using holidays")
            prophet_model = Prophet(weekly_seasonality=True,
                                    daily_seasonality=False)
        pro_df = self._get_prophet_df()
        with suppress_stdout_stderr():
            prophet_model.fit(pro_df)
        
        if self.params.get("auto_weight_season", False):
            # Check the correlation between the dependent variable and the sum
            # of spend and use it to assign a weight to the season trend.
            # print(prophet_model.predict(pro_df[["ds"]]).head())
            try:
                season = prophet_model.predict(pro_df[["ds"]])["yearly"].values
                season = (season - np.mean(season)) / np.std(season)
                spends = np.sum(self.x, axis=1)
                spends = (spends - np.mean(spends)) / np.std(spends)
                corr = np.convolve(season, spends)/spends.size
                corr = np.max(corr)
                season_w = 1
                if corr > 0.2:
                    season_w = 1.16 - 0.8 * corr
                    season_w = 1.0 - 0.8 * corr - 0.4495
                log.info(f"Season auto weight: {season_w:.4f}")
                return prophet_model, season_w
            except KeyError:
                log.warn("No yearly period found. Data range to short.")
                return prophet_model, 1
        else:
            return prophet_model
    
    def set_prophet(self, prophet_model):
        season_w = 1
        if self.params.get("auto_weight_season", False):
            prophet_model, season_w = prophet_model            
        pro_df = self._get_prophet_df()
        preds = prophet_model.predict(pro_df[["ds"]])
        try:
            self.y_season = preds["yearly"].values * season_w
        except KeyError:
            self.y_season = 0
        self.y_hat = self.df[self.params["dep_var"]].values - self.y_season
    
    def _get_prophet_df(self):
        # Prophet needs a dataframe with exactly two columns named 'ds' and 'y'
        date_var = self.params["date_var"]
        dep_var = self.params["dep_var"]
        pro_df = self.df[[date_var, dep_var]]
        pro_df = pro_df.rename(columns={date_var: "ds", dep_var: "y"})
        return pro_df
    
    def get_indeps(self):
        return np.concatenate([self.x, self.c], axis=1)
    
    def get_indeps_cols(self):
        return self.params["ind_var"] + self.params["context"]



class suppress_stdout_stderr:
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    Taken from https://github.com/facebook/prophet/issues/223
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class EvalEffect:
    """
    Helper class while calculating budget opimization.
    """
    def __init__(self, name_idx, xs, spends, intercept):
        self.total_effect = np.sum(xs)
        self.total_budget = np.sum(spends)
        self.roi = self.total_effect / self.total_budget
        self.features = []
        effects = xs / self.total_effect
        for feature, i in name_idx.items():
            feature_item = {
                "name": feature,
                "spend": spends[i],
                "effect": xs[i]
            }
            self.features.append(feature_item)
        
    def merge(self, other):
        """
        Merge this EvalEffect objcet with another [other]
        """
        self.effect_change = (other.total_effect - self.total_effect)\
            / self.total_effect
        self.budget_change = (other.total_budget - self.total_budget)\
            / self.total_budget
        self.allocation = []
        for init_feat, opti_feat in zip(self.features, other.features):
            self.allocation.append(
                {
                    "name": init_feat["name"],
                    "spend": {
                        "optimized": opti_feat["spend"],
                        "current": init_feat["spend"]
                    },
                    "effect": {
                        "optimized": opti_feat["effect"],
                        "current": init_feat["effect"]
                    },
                }
            )
    
    def get_df(self):
        dict_obj = {}
        for item in self.allocation:
            dict_obj[item["name"]] = [
                item["effect"]["current"],
                item["effect"]["optimized"],
                item["spend"]["current"],
                item["spend"]["optimized"],
            ]
        columns = [
            "effect current",
            "effect optimized",
            "spend current",
            "spend optimized"
        ]
        return pd.DataFrame.from_dict(
            dict_obj,
            orient="index",
            columns=columns)