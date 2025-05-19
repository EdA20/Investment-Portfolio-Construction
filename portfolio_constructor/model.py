import json
import logging
import os
from typing import Dict, Union

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from IPython.display import clear_output
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from tqdm import tqdm

from portfolio_constructor import PROJECT_ROOT, TQDM_DISABLE
from portfolio_constructor.custom_metrics import *
from portfolio_constructor.target_markup import position_rotator
from portfolio_constructor.plotter import plot_strategy_performance

# to plot in debug mode use this
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt


def write_log_file(info_to_log: Union[Dict, str] = None):
    logger = logging.getLogger("strategy_logs")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_name = f"trial_{pd.Timestamp.now().strftime('%Y%m%d')}.log"
    log_file = PROJECT_ROOT / f"logs/{file_name}"

    filemode = "a" if log_file.exists() else "w"
    file_handler = logging.FileHandler(log_file, mode=filemode, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d")
    )
    logger.addHandler(file_handler)
    logger.propagate = False

    if isinstance(info_to_log, dict):
        logger.info(json.dumps(info_to_log))
    elif isinstance(info_to_log, str):
        logger.info(info_to_log)


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


def open_random_features_perf_file(path=None):
    if not path:
        json_dir = PROJECT_ROOT / "jsons"
        files = sorted(json_dir.glob("random_features_perf_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError('В папке jsons нет файлов с результатами')
        path = files[0]

    with open(path, "r") as file:
        random_features_pnl = json.load(file)

    random_features_pnl = pd.DataFrame.from_dict(random_features_pnl, orient="index")
    if random_features_pnl.shape[1] > 1:
        if random_features_pnl.shape[1] == 2:
            cols = ["mean_strategy_perf", "std_strategy_perf"]
        else:
            cols = [
                "mean_strategy_perf",
                "std_strategy_perf",
                "mean_mean_outperf",
                "std_mean_outperf",
            ]

        random_features_pnl = random_features_pnl.set_axis(cols, axis=1).sort_values(
            "mean_strategy_perf", ascending=False
        )

    return random_features_pnl


def time_critical_weighting(init_w, returns, k=100, q=0.01, jump_coef=20, fading_factor=21):
    init_w = init_w.copy()
    returns = returns.copy()

    cum_weights = init_w.cumsum()
    alpha = np.log(k) / len(cum_weights)
    time_weighting = np.exp(alpha * cum_weights)

    crit_dates = returns.index[np.where(returns < returns.quantile(q))]
    gamma = np.log(k) / fading_factor
    critical_ws = []
    for crit_date in crit_dates:
        critical_w = init_w.loc[crit_date:]
        critical_w = critical_w.cumsum()
        critical_w = jump_coef * np.exp(-gamma * critical_w)
        critical_ws.append(critical_w)
    critical_weighting = pd.concat(critical_ws, axis=1).fillna(0).max(axis=1)
    weights = pd.concat([time_weighting, critical_weighting], axis=1).fillna(0).sum(axis=1)
    weights = weights / weights.sum()

    return weights


def step_function(x, prob=0.99, convergence_to_prob=100, shift=0):
    base = (1 / prob - 1) ** (-1 / convergence_to_prob)
    value = 1.0 / (1.0 + base ** (-x + shift))
    return value


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

        self.logs = self.__dict__.copy()
        self.logs.pop('feature_info', None)

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

        if mode == 'time_critical':
            cum_weights = data[weight_name].cumsum()
            alpha = np.log(weight_params['k']) / len(cum_weights)
            time_weighting = np.exp(alpha * cum_weights)

            crit_dates = data.index[np.where(data['price_return'] < data['price_return'].quantile(weight_params['q']))]
            gamma = np.log(weight_params['k']) / weight_params['fading_factor']
            critical_ws = []
            for crit_date in crit_dates:
                critical_w = data[weight_name].loc[crit_date:]
                critical_w = critical_w.cumsum()
                critical_w = weight_params['jump_coef'] * np.exp(-gamma * critical_w)
                critical_ws.append(critical_w)
            critical_weighting = pd.concat(critical_ws, axis=1).fillna(0).max(axis=1)

            data[weight_name] = pd.concat([time_weighting, critical_weighting], axis=1).fillna(0).sum(axis=1)

        elif mode == 'rotation_event':
            # пока что не работает
            if weight_params.get('imptnt_obs_lag_reaction') is None:
                raise Exception('pass "imptnt_obs_lag_reaction" arg in "weight_sample_kwargs"')
            if weight_params.get('imptnt_obs_w') is None:
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
            data[weight_name] = data['price_return'].copy()
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

            max_wieght = len(data) * weight_params['obs_frac']
            shifted_returns = data['price_return'].copy()
            data[weight_name] = abs(np.emath.logn(1 + shifted_returns.mean(), 1 + shifted_returns))
            data[weight_name] = np.clip(data[weight_name], 0, max_wieght)
            if weight_params["sqrt_scale"]:
                data[weight_name] = np.sqrt(data[weight_name])

        if weight_smoothing:
            data[weight_name] = data[weight_name]\
                .rolling(window=weight_smoothing['window'], win_type='exponential')\
                .mean(center=weight_smoothing['window'], sym=False, tau=weight_smoothing['win_type_param'])\
                .bfill().ffill()

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
                    train_dates[-self.splitter_kwargs["eval_obs"]: ],
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
        names = ('train', 'test') if self.splitter_kwargs["eval_obs"] <= 0 else ('train', 'val', 'test')
        for chunk_dates in tqdm(dates, desc='position_rotator', disable=TQDM_DISABLE):
            zip_names_dates = zip(names, chunk_dates)

            cum_dates = pd.DatetimeIndex([])
            train_val_test_dates = {}
            for name, dates in zip_names_dates:
                cum_dates = cum_dates.append(dates)
                train_val_test_dates[name] = (dates, cum_dates)

            if self.feature_info is not None:
                idx = max(np.where(self.feature_info.index <= train_val_test_dates['train'][0][-1])[0])
                features = self.feature_info.iloc[idx]

            pools = {}
            subset_dates = {}

            for subset, (split_dates, united_dates) in train_val_test_dates.items():
                target_col = f"{subset}_target"
                weight_col = f"{subset}_weight"
                data_united = data.loc[united_dates.values]

                rotator = position_rotator(data_united["price"], **self.position_rotator_kwargs)
                data_united[target_col] = rotator['target'].copy()

                if subset == 'train' and self.sample_weight_kwargs:
                    obs_weights = self.get_sample_weights(
                        data_united,
                        target_col,
                        weight_col,
                        **self.sample_weight_kwargs,
                    )
                    data_united[weight_col] = obs_weights
                else:
                    data_united[weight_col] = 1

                X = data_united.loc[split_dates, features]
                y = data_united.loc[split_dates, target_col]
                obs_weights = data_united.loc[split_dates, weight_col]
                pools[subset] = dict(data=X, label=y, weight=obs_weights)
                subset_dates[subset] = split_dates

            batches.append((pools, subset_dates))

        clear_output()

        return batches


class Model:

    ALL_MODELS = ('catboost',)

    def __init__(
        self,
        dataset,
        model_kwargs,
    ):

        self.model_kwargs = model_kwargs

        self.logs = self.__dict__.copy()
        self.logs.update(dataset.logs)

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

        if self.logs:
            features = all_train_info.copy()
            features = features.drop_duplicates('from_date')
            features['from_date'] = features['from_date'].astype(str)
            features = features\
                .set_index('from_date')\
                .drop(['mean_train_target', 'mean_test_target', 'till_date', 'n_model'], axis=1)

            features = features.apply(lambda x: list(x[x.notna()].index), axis=1)
            features = features.drop_duplicates().to_dict()
            self.logs['features'] = features

        return all_preds, all_train_info


class Strategy:
    def __init__(self, model, prob_to_weight=True):
        self.prob_to_weight = prob_to_weight

        self.logs = self.__dict__.copy()
        self.logs.update(model.logs)

    def base_strategy_peformance(self, strat_data, preds, plot=False):
        strat_data = strat_data.copy()

        strat_data['preds'] = preds.mean(axis=1)
        strat_data['preds'] = strat_data['preds'].shift()
        strat_data['is_bench_long'] = strat_data['preds'] >= 0.5

        cols = ['price', 'ruonia', 'ruonia_daily', 'preds', 'is_bench_long', 'price_return']
        res = strat_data.loc[:, cols].dropna(subset='preds')

        if self.prob_to_weight:
            res['bench_long_weight'] = res['preds'].apply(
                lambda x: x if x >= 0.5 else 0
            )
        else:
            res["bench_long_weight"] = res["preds"].apply(
                lambda x: 1 if x >= 0.5 else 0
            )

        year_fraq = ((res.index - res.index[0]).days + 1) / (res.index.is_leap_year + 365)
        strat_data['turnover'] = abs(res['bench_long_weight'].diff()).cumsum() / year_fraq

        res['strat_return'] = (
            res['bench_long_weight'] * res['price_return'] +
            (1 - res['bench_long_weight']) * res['ruonia_daily']
        )
        res.loc[res.index[0], ['price_return', 'strat_return']] = 0
        res['strategy_perf'] = (res['strat_return'] + 1).cumprod() * 100
        res['bench_perf'] = (res['price_return'] + 1).cumprod() * 100
        res['outperf'] = res['strategy_perf'] - res['bench_perf']

        self.logs['start_date'] = res.index[0].strftime('%Y-%m-%d')
        self.logs['end_date'] = res.index[-1].strftime('%Y-%m-%d')
        self.logs['strategy_perf'] = res['strategy_perf'].iloc[-1]
        self.logs['bench_perf'] = res['bench_perf'].iloc[-1]
        self.logs['mean_outperf'] = res['outperf'].mean()

        metrics = self.calculate_strategy_metrics(res)
        self.logs['metrics'] = metrics

        if plot:
            plot_strategy_performance(res)

        write_log_file(self.logs)
        output = {
            'res': res,
            'metrics': metrics
        }
        return output

    @staticmethod
    def calculate_strategy_metrics(res):

        structural_shift_dates = ['2014-06-01', '2022-02-24']
        n = np.array([i for i in range(1, len(res) + 1)])
        weights = np.ones(len(res))
        for date in structural_shift_dates:
            shift = np.where(res.index >= date)[0][0]
            weights += 0.5 * step_function(n, prob=0.99, convergence_to_prob=100, shift=shift)
        weights = weights / weights.sum()

        market_outperformance = res['strat_return'] - res['price_return']
        deposit_outperformance = res['strat_return'] - res['ruonia_daily']
        sharpe_ratio_rf = deposit_outperformance.mean() / res['strat_return'].std()
        sharpe_ratio_rm = market_outperformance.mean() / res['strat_return'].std()
        weighted_sharpe_ratio_rf = sum(weights * deposit_outperformance) / res['strat_return'].std()
        weighted_sharpe_ratio_rm = sum(weights * market_outperformance) / res['strat_return'].std()

        drawdown = res['strategy_perf'] / res['strategy_perf'].expanding().max() - 1
        max_drawdown = drawdown.min()

        last_argmax = res['strategy_perf'].expanding().apply(lambda x: x.argmax())
        max_recovery = last_argmax.drop_duplicates().diff().max()

        beta = res[['strat_return', 'price_return']].cov().iloc[0,1] / res['price_return'].var()

        var = res['strat_return'].quantile(0.01)
        cvar = res.loc[res['strat_return'] < var, 'strat_return'].mean()

        metrics = {
            'strategy_perf':  res['strategy_perf'].iloc[-1],
            'bench_perf': res['bench_perf'].iloc[-1],
            'mean_outperf':  res['outperf'].mean(),
            'sharpe_ratio_rf': sharpe_ratio_rf,
            'sharpe_ratio_rm': sharpe_ratio_rm,
            'weighted_sharpe_ratio_rf': weighted_sharpe_ratio_rf,
            'weighted_sharpe_ratio_rm': weighted_sharpe_ratio_rm,
            'max_drawdown': max_drawdown,
            'max_recovery': max_recovery,
            'beta': beta,
            'var': var,
            'cvar': cvar
        }
        return metrics


def strategy_full_cycle(
    data,
    features,
    splitter_kwargs,
    sample_weight_kwargs,
    position_rotator_kwargs,
    model_kwargs,
    prob_to_weight
):
    dataset = Dataset(
        splitter_kwargs,
        sample_weight_kwargs,
        position_rotator_kwargs,
    )
    batches = dataset.get_batches(data, features)

    model = Model(
        dataset,
        model_kwargs,
    )
    preds, train_info = model.get_predictions(batches)

    strategy = Strategy(
        model,
        prob_to_weight=prob_to_weight
    )
    strat_data = data.loc[:, ['price', 'price_return', 'ruonia', 'ruonia_daily']].copy()
    output = strategy.base_strategy_peformance(strat_data, preds)

    return output


if __name__ == "__main__":
    df = read_logger()
    a=1
