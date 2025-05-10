from utils import PROJECT_ROOT, TQDM_DISABLE

import os
import json
import logging

import numpy as np
import pandas as pd

from utils.feature_generator import data_generator, position_rotator
from utils.custom_metrics import *
from utils.strategy_plots import plot_startegy_performance

from typing import Union, Dict
from IPython.display import display, clear_output
from tqdm import tqdm
from catboost import Pool, CatBoostClassifier
from sktime.split import SlidingWindowSplitter, ExpandingWindowSplitter


def write_log_file(info_to_log: Union[Dict, str] = None):
    file_name = f"strategy_logger_{pd.Timestamp.now().strftime('%Y%m%d')}.log"
    is_file_name = list(
        map(lambda x: x == file_name, os.listdir(PROJECT_ROOT / "logs"))
    )
    if any(is_file_name):
        filemode = "a"
    else:
        filemode = "w"

    logging.basicConfig(
        filename=PROJECT_ROOT / f"logs/{file_name}",
        level=logging.INFO,
        datefmt="%Y-%m-%d",
        filemode=filemode,
        format="%(asctime)s - %(message)s",
    )
    if isinstance(info_to_log, dict):
        logging.info(json.dumps(info_to_log))

    for handler in logging.root.handlers:
        logging.getLogger().removeHandler(handler)


def read_logger(path: str = None):
    paths = [path] if path else os.listdir(PROJECT_ROOT / "logs")

    df_logs = []
    for path in paths:
        df_log = []
        with open(PROJECT_ROOT / "logs" / path, "r") as file:
            for line in file:
                df_log.append(json.loads(line.strip().split("- ")[1]))

        df_log = pd.DataFrame(df_log)
        df_log["run_date"] = pd.to_datetime(
            path.split("_")[-1].split(".")[0], format="%Y%m%d"
        )
        df_logs.append(df_log)

    df_logs = pd.concat(df_logs, ignore_index=True)
    df_logs = df_logs.sort_values("run_date", ascending=False, ignore_index=True)

    return df_logs


def open_random_features_pnl_file(path=None):
    if not path:
        path = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}.json"

    with open(path, "r") as file:
        random_features_pnl = json.load(file)

    random_features_pnl = pd.DataFrame.from_dict(random_features_pnl, orient="index")
    if random_features_pnl.shape[1] > 1:
        if random_features_pnl.shape[1] == 2:
            cols = ["pnl_mean", "pnl_std"]
        else:
            cols = [
                "pnl_mean",
                "pnl_std",
                "mean_median_perf_delta",
                "mean_std_perf_delta",
            ]

        random_features_pnl = random_features_pnl.set_axis(cols, axis=1).sort_values(
            "pnl_mean", ascending=False
        )

    return random_features_pnl


class Dataset:
    def __init__(
        self,
        splitter_kwargs,
        sample_weight_kwargs,
        position_rotator_kwargs,
        feature_info=None,
    ):
        self.splitter_kwargs = splitter_kwargs
        self.sample_weight_kwargs = sample_weight_kwargs
        self.position_rotator_kwargs = position_rotator_kwargs
        self.feature_info = feature_info

    @staticmethod
    def propect_value_func(x, ind_estim=1, neg_bias=0.9, ext_mult=5):
        value = x**ind_estim if x >= 0 else -neg_bias * (-x) ** ind_estim
        if abs(x) >= 5:
            value *= ext_mult
        else:
            value = 1

        return value

    @staticmethod
    def get_sample_weights(
        data,
        rotation_column,
        weight_name="obs_weight",
        weight_params=None,
        weight_smoothing=None,
    ):
        if weight_params is None:
            raise Exception('Pass "weight_params" argument')
        data = data.copy()

        data[weight_name] = 1
        mode = list(weight_params.keys())[0]
        weight_params = weight_params[mode]

        if mode == "rotation_event":
            if weight_params.get("imptnt_obs_lag_reaction") is None:
                raise Exception(
                    'pass "imptnt_obs_lag_reaction" arg in "weight_sample_kwargs"'
                )
            if weight_params.get("imptnt_obs_w") is None:
                raise Exception('pass "imptnt_obs_w" arg in "weight_sample_kwargs"')

            reaction_days = np.arange(
                1, 1 + weight_params["imptnt_obs_lag_reaction"]
            ).reshape(-1, 1)
            rotation_period_idx = (
                np.where(data[rotation_column].notna())[0] + reaction_days
            ).ravel()
            rotation_period_idx = rotation_period_idx[rotation_period_idx < len(data)]

            rotation_period = data.index[rotation_period_idx]
            rotation_period = rotation_period[rotation_period > "2011-01-01"]

            data.loc[rotation_period, weight_name] = weight_params["imptnt_obs_w"]

        elif mode == "dramatic_return_event":
            if weight_params.get("return_mult") is None:
                raise Exception('pass "return_mult" arg for "weight_sample_kwargs"')
            if weight_params.get("return_thresh") is None:
                raise Exception('pass "return_thresh" arg for "weight_sample_kwargs"')

            data[weight_name] = data["price_return"].shift(-1).ffill()
            data[weight_name] = data[weight_name].apply(
                lambda x: x * weight_params["return_mult"]
                if (x <= -weight_params["return_thresh"])
                or (x >= weight_params["return_thresh"])
                else 0.01
            )
            data[weight_name] = abs(data[weight_name]) * 100

        elif mode == "compare_mean_return":
            if weight_params.get("obs_frac") is None:
                raise Exception('pass "obs_frac" arg for "weight_sample_kwargs"')
            if weight_params.get("sqrt_scale") is None:
                raise Exception('pass "sqrt_scale" arg for "weight_sample_kwargs"')

            max_wieght = len(data) * weight_params["obs_frac"]
            shifted_returns = data["price_return"].shift(-1).ffill()
            data[weight_name] = abs(
                np.emath.logn(1 + shifted_returns.mean(), 1 + shifted_returns)
            )
            data[weight_name] = np.clip(data[weight_name], 0, max_wieght)
            if weight_params["sqrt_scale"]:
                data[weight_name] = np.sqrt(data[weight_name])

        if weight_smoothing.get("win_type") == "exp":
            data[weight_name] = (
                data[weight_name]
                .rolling(window=weight_smoothing["window"], win_type="exponential")
                .mean(
                    center=weight_smoothing["window"],
                    sym=False,
                    tau=weight_smoothing["win_type_param"],
                )
                .bfill()
                .ffill()
            )

        elif weight_smoothing.get("win_type") == "gauss":
            data[weight_name] = (
                data[weight_name]
                .rolling(
                    window=weight_smoothing["window"], center=True, win_type="gaussian"
                )
                .mean(std=weight_smoothing["win_type_param"])
                .bfill()
                .ffill()
            )

        return data[weight_name]

    def get_batches(self, data, features=None):
        data = data.copy()

        if features:
            features_idx = np.where(data.columns.isin(features))[0]
            if len(features_idx) != len(features):
                absent_features = list(
                    set(features) - set(data.iloc[:, features_idx].columns)
                )
                raise Exception(f"в data нет фичей {absent_features}")

        if (
            self.splitter_kwargs["initial_window"]
            and self.splitter_kwargs["window_length"]
        ) or (
            not self.splitter_kwargs["initial_window"]
            and not self.splitter_kwargs["window_length"]
        ):
            raise Exception(
                "initial_window исп-ся для расширящегося окна, window_length - для скользящего. Нужно выбрать что-то одно"
            )

        if self.splitter_kwargs["window_length"]:
            splitter = SlidingWindowSplitter(
                fh=self.splitter_kwargs["forecast_horizon"],
                window_length=self.splitter_kwargs["window_length"],
                step_length=self.splitter_kwargs["step_length"],
            )
        else:
            splitter = ExpandingWindowSplitter(
                fh=self.splitter_kwargs["forecast_horizon"],
                initial_window=self.splitter_kwargs["initial_window"],
                step_length=self.splitter_kwargs["step_length"],
            )

        dates = list(splitter.split_loc(data.index))
        oob_test_dates = pd.DatetimeIndex(
            set(data.index[-self.splitter_kwargs["step_length"] + 1 :])
            - set(dates[-1][1])
        )
        oob_test_dates.name = "date"
        dates[-1] = (dates[-1][0], dates[-1][1].append(oob_test_dates))
        if self.splitter_kwargs["eval_obs"] > 1:
            dates = [
                (
                    train_dates[: -self.splitter_kwargs["eval_obs"]],
                    train_dates[-self.splitter_kwargs["eval_obs"] :],
                    test_dates,
                )
                for train_dates, test_dates in dates
            ]
        elif 0 < self.splitter_kwargs["eval_obs"] < 1:
            dates = [
                (
                    train_dates[
                        : -int(len(train_dates) * self.splitter_kwargs["eval_obs"])
                    ],
                    train_dates[
                        -int(len(train_dates) * self.splitter_kwargs["eval_obs"]) :
                    ],
                    test_dates,
                )
                for train_dates, test_dates in dates
            ]

        batches = []
        # batches = []
        # names = ('train', 'test') if eval_obs <= 0 else ('train', 'val', 'test')
        # train_val_test_dates = {}
        # for splitted_dates in tqdm(dates, desc='position_rotator', disable=TQDM_DISABLE):
        #     zip_names_dates = zip(names, splitted_dates)
        #     cum_dates = pd.DatetimeIndex([])
        #     temp_dct = {}
        #     for name, dates in zip_names_dates:
        #         cum_dates.append(dates)
        #         temp_dct[name] = (dates, cum_dates)
        #         train_val_test_dates.update

        for splitted_dates in tqdm(
            dates, desc="position_rotator", disable=TQDM_DISABLE
        ):
            if self.splitter_kwargs["eval_obs"] > 0:
                train, val, test = (
                    splitted_dates[0],
                    splitted_dates[1],
                    splitted_dates[2],
                )
                train_val = train.append(val)
                train_val_test = train_val.append(test)

                train_val_test_dates = {
                    "train": (train, train),
                    "val": (val, train_val),
                    "test": (test, train_val_test),
                }
            else:
                train, test = splitted_dates[0], splitted_dates[1]
                train_test = train.append(test)

                train_val_test_dates = {
                    "train": (train, train),
                    "test": (test, train_test),
                }

            if self.feature_info is not None:
                idx = max(np.where(self.feature_info.index <= train[-1])[0])
                features = self.feature_info.iloc[idx]

            pools = {}
            subset_dates = {}

            for subset, (splitted_dts, united_dts) in train_val_test_dates.items():
                data[f"{subset}_target"] = np.nan
                rotator = position_rotator(
                    data.loc[united_dts, "price"], **self.position_rotator_kwargs
                )
                rotator = rotator.rename({"action": f"{subset}_target"}, axis=1)
                data[f"{subset}_target"] = rotator[f"{subset}_target"].copy()

                data[f"{subset}_weight"] = np.nan
                if self.sample_weight_kwargs:
                    obs_weights = self.get_sample_weights(
                        data.loc[united_dts],
                        f"{subset}_target",
                        f"{subset}_weight",
                        **self.sample_weight_kwargs,
                    )
                    data[f"{subset}_weight"] = obs_weights
                else:
                    data[f"{subset}_weight"] = 1
                data[f"{subset}_target"] = (
                    data[f"{subset}_target"].ffill() == "buy"
                ).astype(int)

                X = data.loc[splitted_dts, features]
                y = data.loc[splitted_dts, f"{subset}_target"]
                obs_weights = data.loc[splitted_dts, f"{subset}_weight"]
                pools[subset] = dict(data=X, label=y, weight=obs_weights)
                subset_dates[subset] = splitted_dts

            batches.append((pools, subset_dates))

        clear_output()

        return batches


class StrategyModeller(Dataset):
    ALL_MODELS = ("catboost",)

    def __init__(
        self,
        splitter_kwargs,
        sample_weight_kwargs,
        position_rotator_kwargs,
        model_kwargs,
        prob_to_weight=True,
        logging=True,
        feature_info=None,
    ):
        super().__init__(
            splitter_kwargs, sample_weight_kwargs, position_rotator_kwargs, feature_info
        )
        self.model_kwargs = model_kwargs
        self.prob_to_weight = prob_to_weight

        if logging:
            self.log_collector = self.__dict__.copy()
            self.log_collector.pop("feature_info", None)
        else:
            self.log_collector = None

        self.output = {}

        os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
        os.makedirs(PROJECT_ROOT / "jsons", exist_ok=True)

    def get_model_output(self, batches, **kwargs):
        model_kwargs = self.model_kwargs.copy()
        model_kwargs.pop("n_models")
        model_name = model_kwargs.pop("model_name")

        dct_preds = {}

        if model_name == "catboost":
            train_infos = []
            if len(batches[0]) == 2:
                for pools, subset_dates in tqdm(
                    batches, desc="batches", disable=TQDM_DISABLE
                ):
                    pools = pools.copy()
                    for subset, subset_data in pools.items():
                        pools[subset] = Pool(**subset_data)

                    model = CatBoostClassifier(**model_kwargs)
                    model.fit(pools["train"], verbose=False)
                    preds = model.predict_proba(pools["test"])[:, 1]
                    date_preds = dict(zip(list(subset_dates["test"]), list(preds)))
                    dct_preds.update(date_preds.copy())

                    mean_train_target = pools["train"].get_label().mean()
                    mean_test_target = pools["test"].get_label().mean()
                    first_test_date = subset_dates["test"][0]
                    last_test_date = subset_dates["test"][-1]

                    features = pools["train"].get_feature_names()
                    columns = [
                        "from_date",
                        "till_date",
                        "mean_train_target",
                        "mean_test_target",
                    ] + list(features)
                    train_info = [
                        first_test_date,
                        last_test_date,
                        mean_train_target,
                        mean_test_target,
                        *model.get_feature_importance(),
                    ]
                    train_info = pd.DataFrame([train_info], columns=columns)
                    train_infos.append(train_info)
            else:
                for pools, subset_dates in tqdm(
                    batches, desc="batches", disable=TQDM_DISABLE
                ):
                    pools = pools.copy()
                    for subset, subset_data in pools.items():
                        pools[subset] = Pool(**subset_data)

                    model = CatBoostClassifier(
                        loss_function=FocalLossObjective(),
                        eval_metric=FocalLossMetric(),
                        **model_kwargs,
                    )
                    model.fit(
                        pools["train"],
                        use_best_model=True,
                        eval_set=pools["val"],
                        verbose=False,
                    )
                    preds = model.predict_proba(pools["test"])[:, 1]
                    dct_preds.update(dict(zip(subset_dates["test"], preds)))

                    mean_train_target = pools["train"].get_label().mean()
                    mean_test_target = pools["test"].get_label().mean()
                    first_test_date = subset_dates["test"][0]
                    last_test_date = subset_dates["test"][-1]

                    features = pools["train"].get_feature_names()
                    columns = [
                        "from_date",
                        "till_date",
                        "mean_train_target",
                        "mean_test_target",
                    ] + list(features)
                    train_info = [
                        first_test_date,
                        last_test_date,
                        mean_train_target,
                        mean_test_target,
                        *model.get_feature_importance(),
                    ]
                    train_info = pd.DataFrame([train_info], columns=columns)
                    train_infos.append(train_info)

            preds = pd.DataFrame.from_dict(dct_preds, orient="index", columns=["preds"])
            train_infos = pd.concat(train_infos, ignore_index=True)

        else:
            raise Exception(
                f"Модель {model_name} не засетаплена, исп-те название из перечня: {self.ALL_MODELS}"
            )

        return preds, train_infos, model

    def get_predictions(self, batches):
        all_preds = []
        all_train_info = []

        seed = self.model_kwargs["random_state"]
        rng = range(seed, seed + self.model_kwargs["n_models"])

        for i in tqdm(rng, desc="total_models", disable=TQDM_DISABLE):
            self.model_kwargs["random_state"] = i
            preds, train_info, model = self.get_model_output(batches)
            train_info["n_model"] = i - seed + 1

            all_preds.append(preds)
            all_train_info.append(train_info)

        all_preds = pd.concat(all_preds, axis=1).set_axis(
            [f"preds_seed_{i}" for i in rng], axis=1
        )
        all_train_info = pd.concat(all_train_info, ignore_index=True)

        self.output["train_info"] = all_train_info.copy()
        if self.log_collector:
            all_train_info = all_train_info.drop_duplicates("from_date")
            all_train_info["from_date"] = all_train_info["from_date"].astype(str)
            features = all_train_info.set_index("from_date").drop(
                ["mean_train_target", "mean_test_target", "till_date", "n_model"],
                axis=1,
            )
            features = features.apply(
                lambda x: list(x[x.notna()].index), axis=1
            ).to_dict()
            self.log_collector["features"] = features

        return all_preds, all_train_info

    def base_strategy_peformance(self, strat_data, preds, plot=False):
        strat_data = strat_data.copy()

        strat_data["preds"] = preds.mean(axis=1)
        strat_data["turnover"] = abs(
            strat_data["preds"] - strat_data["preds"].shift()
        ).cumsum() / (((strat_data.index - strat_data.index[0]).days + 1) / 365)
        strat_data["preds"] = strat_data["preds"].shift()
        strat_data["is_bench_long"] = strat_data["preds"] >= 0.5

        cols = [
            "price",
            "ruonia",
            "ruonia_daily",
            "preds",
            "is_bench_long",
            "price_return",
            "turnover",
        ]
        res = strat_data.loc[:, cols].dropna(subset="preds")

        if self.prob_to_weight:
            res["bench_long_weight"] = res["preds"].apply(
                lambda x: x if x >= 0.5 else 0
            )
        else:
            res["bench_long_weight"] = res["preds"].apply(
                lambda x: 1 if x >= 0.5 else 0
            )

        res["strat_return"] = (
            res["bench_long_weight"] * res["price_return"]
            + (1 - res["bench_long_weight"]) * res["ruonia_daily"]
        )
        res.iloc[0, np.where(res.columns == "start_return")[0]] = 0

        res.iloc[0, np.where(res.columns.isin(["price_return", "strat_return"]))[0]] = 0
        res["strategy_perf"] = (res["strat_return"] + 1).cumprod() * 100
        res["bench_perf"] = (res["price_return"] + 1).cumprod() * 100
        res["perf_delta"] = res["strategy_perf"] - res["bench_perf"]

        test_subset_idx = np.where(
            res.index.isin(self.output["train_info"]["from_date"].unique())
        )
        res["test_subset_idx"] = 0
        res.loc[res.index[test_subset_idx], "test_subset_idx"] = 1
        res["test_subset_idx"] = res["test_subset_idx"].cumsum()
        test_subset_pnl = res.groupby(res["test_subset_idx"])[
            ["price_return", "strat_return"]
        ].apply(lambda x: (1 + x).prod() - 1)
        test_subset_outperf = (
            test_subset_pnl["strat_return"] - test_subset_pnl["price_return"]
        )

        n_test_subsets = np.array([i for i in range(1, len(test_subset_outperf) + 1)])
        weight_function_1 = lambda x, power: x**power / sum(x**power)
        weight_function_2 = (
            lambda x, base: 1.0
            / (1.0 + base ** (-x + 36.0))
            / sum(1.0 / (1.0 + base ** (-x + 36.0)))
        )
        weights = weight_function_2(n_test_subsets, 1.1) + weight_function_1(
            n_test_subsets, 1 / 2
        ) / sum(
            weight_function_2(n_test_subsets, 1.1)
            + weight_function_1(n_test_subsets, 1 / 2)
        )
        mean_test_subset_outperf = 100 * test_subset_outperf.mean()
        weighted_test_subset_outperf = 100 * sum(test_subset_outperf * weights)
        weighted_test_sharpe = weighted_test_subset_outperf / (
            res["strategy_perf"].std() / 100
        )

        print(
            "ML Strategy Outperformance:",
            res["strategy_perf"].iloc[-1] - res["bench_perf"].iloc[-1],
        )
        print("Median Performance Delta:", res["perf_delta"].median())
        print("Mean Outperformance Test Folds:", mean_test_subset_outperf)
        print("Weighted Outperformance Test Folds:", weighted_test_subset_outperf)
        print(
            "Weighted Outperformance Sharpe Metric:", weighted_test_sharpe, end="\n\n"
        )

        if self.log_collector:
            self.log_collector["start_date"] = res.index[0].strftime("%Y-%m-%d")
            self.log_collector["random_state"] = self.model_kwargs["random_state"]
            self.log_collector["strategy_pnl"] = res["strategy_perf"].iloc[-1]
            self.log_collector["bench_pnl"] = res["bench_perf"].iloc[-1]
            self.log_collector["median_perf_delta"] = res["perf_delta"].median()
            self.log_collector["mean_test_subset_outperf"] = mean_test_subset_outperf
            self.log_collector["weighted_test_subset_outperf"] = (
                weighted_test_subset_outperf
            )
            self.log_collector["weighted_test_sharpe_metric"] = weighted_test_sharpe
            write_log_file(self.log_collector)

        if plot:
            plot_startegy_performance(res)

        output = {
            "data": strat_data,
            "res": res,
            "weighted_test_sharpe": weighted_test_sharpe,
        }
        self.output.update(output)

        return self.output


if __name__ == "__main__":
    df = read_logger()
    df["features"] = df["features"].apply(lambda x: tuple(x))
    group_median_perf_delta = (
        df.groupby(["run_date", "features"])["median_perf_delta"].mean().reset_index()
    )
    df = pd.merge(
        df,
        group_median_perf_delta,
        how="left",
        on=["run_date", "features"],
        suffixes=("", "_mean"),
    )
    max_median_perf_delta = df.loc[
        df.groupby("run_date")["median_perf_delta_mean"].idxmax()
    ].sort_values("run_date", ascending=False)
    last_trial_best_features = max_median_perf_delta["features"].iloc[0]
    a = 1
