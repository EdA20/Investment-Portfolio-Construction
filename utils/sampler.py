from utils import PROJECT_ROOT, ENDOG_DATA_FILE, ALL_DATA_FILES, ALL_DATA_PATH

import os
import json
import optuna

import pandas as pd
import numpy as np

from random import shuffle
from IPython.display import clear_output, display
from tqdm import tqdm

from utils.model import StrategyModeller, write_log_file
from utils.plotter import *


def get_json_path():
    base_name = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}"
    suffix = ""
    i = 1
    while True:
        file_path = PROJECT_ROOT / f"{base_name}{suffix}.json"
        if not file_path.exists():
            break
        suffix = f"_{i}"
        i += 1

    return str(file_path)


class SampleStrategy(StrategyModeller):
    def __init__(
        self,
        splitter_kwargs,
        sample_weight_kwargs,
        position_rotator_kwargs,
        model_kwargs,
        prob_to_weight=True,
        logging=True,
        feature_importance_info=None,
    ):
        super().__init__(
            splitter_kwargs,
            sample_weight_kwargs,
            position_rotator_kwargs,
            model_kwargs,
            prob_to_weight,
            logging,
            feature_importance_info,
        )

    # перебор по сиду
    def sample_random_seed_strategy(self, size=100):
        random_seed_pnls = {0: 0}

        for i in tqdm(range(size)):
            df_random_seed_pnls = (
                pd.DataFrame.from_dict(random_seed_pnls, orient="index")
                .reset_index()
                .set_axis(["seed", "PnL"], axis=1)
            )
            display(df_random_seed_pnls)

            output = self.startegy_performance(seed=i)
            res = output["res"].copy()

            random_seed_pnls[i] = res["strategy_perf"].iloc[-1]

            clear_output(wait=True)

        df_random_seed_pnls = (
            pd.DataFrame.from_dict(random_seed_pnls, orient="index")
            .reset_index()
            .set_axis(["seed", "PnL"], axis=1)
        )

        return df_random_seed_pnls

    # перебор по дате начала
    def sample_shifted_strategy(
        self,
        choose_before="2014-01-01",
        init_window_resize=0,
        size=100,
        plot_with_date_normalization=True,
    ):
        dates_before = self.data.index[self.data.index < choose_before]
        dates_idx = np.linspace(0, len(dates_before) - 1, size, dtype=int)
        from_dates = dates_before[dates_idx]

        init_w = self.splitter_kwargs["initial_window"]
        if init_window_resize > 0:
            new_init_windows = init_window_resize - dates_idx
        else:
            new_init_windows = np.ones(len(dates_idx), dtype=int) * init_w

        dates_windows = list(zip(from_dates, new_init_windows))
        sample_strategy_returns = {}
        pnl = {from_dates[0]: (0, 0)}
        for from_date, new_init_w in tqdm(dates_windows):
            df_pnl = (
                pd.DataFrame.from_dict(pnl, orient="index")
                .reset_index()
                .set_axis(["start_date", "PnL", "median_perf_delta"], axis=1)
            )
            display(df_pnl)

            self.splitter_kwargs["initial_window"] = int(new_init_w)
            output = self.startegy_performance(data=self.data.loc[from_date:])
            res = output["res"].copy()
            sample_strategy_returns[from_date] = res["strat_return"].copy()
            pnl[from_date] = (res["strategy_perf"].iloc[-1], res["perf_delta"].median())

            clear_output(wait=True)

        sample_strategy_returns = pd.DataFrame(sample_strategy_returns)
        if plot_with_date_normalization:
            begin_date = (
                sample_strategy_returns.loc[:, sample_strategy_returns.columns.max()]
                .notna()
                .idxmax()
            )
            sample_strategy_returns = sample_strategy_returns.loc[begin_date:].copy()
            plot_shifted_strategy_with_benchmark(self.data, sample_strategy_returns)

        return sample_strategy_returns, pnl

    # перебор по переменным
    def random_features_sampling(
        self,
        all_features,
        min_amount_features=7,
        max_amount_features=15,
        pnl_threshold=300,
        n_trials=1000,
        seed_sampling=None,
        path=None,
    ):
        if not path:
            path = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
            is_file = list(map(lambda x: x == path, os.listdir("../jsons")))
            i = 1
            while any(is_file):
                path = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}_{i}.json"
                is_file = list(map(lambda x: x == path, os.listdir("../jsons")))
                i += 1

        ns_features_to_select = np.random.choice(
            range(min_amount_features, max_amount_features + 1), size=n_trials
        )
        random_features_pnl = {}
        counter = dict(zip(all_features, np.zeros(len(all_features))))

        for n_features_to_select in tqdm(ns_features_to_select):
            features = list(
                np.random.choice(all_features, replace=False, size=n_features_to_select)
            )

            res_seed = []
            median_perf_delta_seed = []
            std_perf_delta_seed = []
            for i in range(1, seed_sampling + 1):
                output = self.startegy_performance(features=features, seed=i)
                res = output["res"].copy()
                if res["strategy_perf"].iloc[-1] < pnl_threshold:
                    break

                res_seed.append(res["strategy_perf"].iloc[-1])
                median_perf_delta_seed.append(res["perf_delta"].median())
                std_perf_delta_seed.append(res["perf_delta"].std())

            if len(res_seed) < seed_sampling:
                continue

            random_features_pnl[f"{tuple(features)}"] = (
                np.mean(res_seed),
                np.std(res_seed),
                np.mean(median_perf_delta_seed),
                np.mean(std_perf_delta_seed),
            )
            for feature in features:
                counter[feature] += 1
                write_log_file(counter)

            json_data = json.dumps(random_features_pnl, indent=4)
            with open(path, "w") as file:
                file.write(json_data)

            clear_output(wait=True)

        return random_features_pnl

    @staticmethod
    def get_good_features(counter, max_amount_features, fraction=0.1):
        prob_features_to_choose = {
            key: value / sum(counter.values()) for key, value in counter.items()
        }
        good_features = np.random.choice(
            list(prob_features_to_choose.keys()),
            replace=False,
            size=int(max_amount_features * fraction),
            p=list(prob_features_to_choose.values()),
        )
        good_features = list(good_features)

        return good_features

    # перебор по переменным
    def random_features_metric_sampling(
        self,
        all_features,
        min_amount_features=10,
        max_amount_features=40,
        n_trials=1000,
        n_seed=1,
        path=None,
    ):
        if not path:
            path = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
            is_file = list(map(lambda x: x == path, os.listdir(PROJECT_ROOT / "jsons")))
            i = 1
            while any(is_file):
                path = path.split(".")[0] + f"_{i}.json"
                is_file = list(
                    map(lambda x: x == path, os.listdir(PROJECT_ROOT / "jsons"))
                )
                i += 1

        all_features = list(all_features)
        shuffle(all_features)

        ns_features_to_select = np.random.choice(
            range(min_amount_features, max_amount_features + 1), size=n_trials
        )
        random_features_pnl = {}
        feature_metric_accum = dict(zip(all_features, np.zeros(len(all_features))))
        exp_feature_metric_accum = {
            key: np.exp(value) for key, value in feature_metric_accum.items()
        }
        feature_softmax = {
            key: value / sum(exp_feature_metric_accum.values())
            for key, value in exp_feature_metric_accum.items()
        }

        for n_features_to_select in tqdm(ns_features_to_select, desc="n_trials"):
            features = np.random.choice(
                list(feature_softmax.keys()),
                replace=False,
                size=n_features_to_select,
                p=list(feature_softmax.values()),
            )
            features = list(features)
            res_seed = []
            median_perf_delta_seed = []
            std_perf_delta_seed = []

            for i in range(1, n_seed + 1):
                output = self.startegy_performance(features=features, seed=i)
                res = output["res"].copy()

                n = np.array([i for i in range(1, len(res) + 1)])
                weights = n**4.0 / sum(n**4.0)

                res["weighted_perf_delta"] = res["perf_delta"] * weights
                metric = (
                    res["weighted_perf_delta"].mean()
                    / (res["strategy_perf"].std() / 100)
                    * 10
                )
                for feature in features:
                    feature_metric_accum[feature] += metric
                exp_feature_metric_accum = {
                    key: np.exp(value) for key, value in feature_metric_accum.items()
                }
                feature_softmax = {
                    key: value / sum(exp_feature_metric_accum.values())
                    for key, value in exp_feature_metric_accum.items()
                }

                res_seed.append(res["strategy_perf"].iloc[-1])
                median_perf_delta_seed.append(res["perf_delta"].median())
                std_perf_delta_seed.append(res["perf_delta"].std())

            if len(res_seed) < n_seed:
                continue

            random_features_pnl[f"{tuple(features)}"] = (
                np.mean(res_seed),
                np.std(res_seed),
                np.mean(median_perf_delta_seed),
                np.mean(std_perf_delta_seed),
            )

            json_data = json.dumps(random_features_pnl, indent=4)
            with open(PROJECT_ROOT / path, "w") as file:
                file.write(json_data)

            clear_output(wait=True)

        return random_features_pnl

    # перебор по переменным
    def random_features_group_sampling(
        self,
        all_features,
        min_amount_features=0,
        max_amount_features=4,
        pnl_thresh=300,
        perf_delta_median_thresh=0,
        n_passed_hard_counter_thresh=10,
        n_trials=1000,
        n_seed=5,
        path=None,
    ):
        if not path:
            path = f"jsons/random_features_pnl_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
            is_file = list(map(lambda x: x == path, os.listdir(PROJECT_ROOT / "jsons")))
            i = 1
            while any(is_file):
                path = path.split(".")[0] + f"_{i}.json"
                is_file = list(
                    map(lambda x: x == path, os.listdir(PROJECT_ROOT / "jsons"))
                )
                i += 1

        all_features = list(all_features)
        shuffle(all_features)

        exog_data_name_path_dct = dict(zip(ALL_DATA_FILES, ALL_DATA_PATH))
        feature_groups = {}
        all_selected_cols = []

        for name, path in exog_data_name_path_dct.items():
            temp = pd.read_excel(path, index_col=[0])
            grouped_cols = []

            for col in temp.columns:
                features = list(filter(lambda x: x.startswith(col), all_features))
                grouped_cols.extend(features)
            feature_groups[name] = grouped_cols.copy()
            all_selected_cols.extend(grouped_cols.copy())
        feature_groups["other"] = list(set(all_features) - set(all_selected_cols))

        per_feature_group_amount = {}
        for group in feature_groups.keys():
            if group in ENDOG_DATA_FILE:
                per_feature_group_amount[group] = {
                    "min_amount_features": max_amount_features,
                    "max_amount_features": max_amount_features * 2,
                }
            else:
                per_feature_group_amount[group] = {
                    "min_amount_features": min_amount_features,
                    "max_amount_features": max_amount_features,
                }

        random_features_pnl = {}
        hard_counter = dict(zip(all_features, np.zeros(len(all_features))))
        good_features = []
        n_passed_hard_counter = 1
        good_feature_trials = 0
        count_n_passed_counter = {"simple": 0, "hard": 0}

        ns_selected_features_in_group = []
        for group, amounts in per_feature_group_amount.items():
            group_random_amounts = np.random.choice(
                range(
                    amounts["min_amount_features"], amounts["max_amount_features"] + 1
                ),
                size=n_trials,
            )
            group_random_amounts = group_random_amounts[:, None].copy()
            ns_selected_features_in_group.append(group_random_amounts)
        ns_selected_features_in_group = np.hstack(ns_selected_features_in_group)

        for n_selected_features_in_group in tqdm(
            ns_selected_features_in_group, desc="n_trials"
        ):
            features = []
            if len(good_features) > 0:
                good_feature_trials += 1
                features = [good_features]
                if good_feature_trials == 10:
                    good_features = []

            for i, (exog_feature, generated_features) in enumerate(
                feature_groups.items()
            ):
                selected_generated_features = list(
                    np.random.choice(
                        generated_features,
                        size=n_selected_features_in_group[i],
                        replace=False,
                    )
                )
                features.extend(selected_generated_features.copy())

            res_seed = []
            median_perf_delta_seed = []
            std_perf_delta_seed = []

            for i in range(1, n_seed + 1):
                print("count_n_passed_counter:", count_n_passed_counter)

                output = self.startegy_performance(features=features, seed=i)
                res = output["res"].copy()
                if (
                    res["strategy_perf"].iloc[-1] < pnl_thresh
                    or res["perf_delta"].median() < perf_delta_median_thresh
                ):
                    break

                res_seed.append(res["strategy_perf"].iloc[-1])
                median_perf_delta_seed.append(res["perf_delta"].median())
                std_perf_delta_seed.append(res["perf_delta"].std())

            if len(res_seed) < n_seed:
                continue

            random_features_pnl[f"{tuple(features)}"] = (
                np.mean(res_seed),
                np.std(res_seed),
                np.mean(median_perf_delta_seed),
                np.mean(std_perf_delta_seed),
            )
            for feature in features:
                hard_counter[feature] += 1
            n_passed_hard_counter += 1

            if n_passed_hard_counter % n_passed_hard_counter_thresh == 0:
                good_features = self.get_good_features(
                    hard_counter, max_amount_features, fraction=1
                )
                good_feature_trials = 0
                count_n_passed_counter["hard"] += 1

            json_data = json.dumps(random_features_pnl, indent=4)
            with open(PROJECT_ROOT / path, "w") as file:
                file.write(json_data)

            clear_output(wait=True)

        return random_features_pnl

    def optuna_optimization_sampling(
        self,
        all_features,
        metric="min_delta",
        n_trials=1000,
        n_seed=3,
        json_path=None,
    ):
        _all_metrics = ["weighted_test_sharpe", "min_delta"]
        if metric not in _all_metrics:
            raise Exception(
                f'No such metric - "{metric}", available metrics: {_all_metrics}'
            )

        all_features = list(all_features)
        shuffle(all_features)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.NSGAIISampler(
                crossover_prob=0.7,
                mutation_prob=0.2,
            ),
        )
        study.optimize(
            lambda trial: self._objective(
                trial, metric, all_features, n_seed, json_path
            ),
            n_trials=n_trials,
        )
        print("Best Features:", study.best_trial.user_attrs["features"])
        print("Best Metric Value:", study.best_value)

    def _objective(self, trial, metric, all_features, n_seed, json_path):
        if json_path is None:
            json_path = get_json_path()

        is_feature_used = [
            trial.suggest_categorical(name, [True, False]) for name in all_features
        ]
        opted_features = [f for f, m in zip(all_features, is_feature_used) if m]

        mean_metric_value = 0
        seed_res = {}
        for i in range(1, n_seed + 1):
            output = self.startegy_performance(features=opted_features, seed=i)
            if metric == "weighted_test_sharpe":
                metric_value = output["weighted_test_sharpe"].copy()
            elif metric == "min_delta":
                metric_value = output["res"]["perf_delta"].min()
            else:
                raise Exception("Как ты проскочил мимо первого экспешена?!?!")

            mean_metric_value += (1 / n_seed) * metric_value
            seed_res[i] = {
                metric: metric_value,
                "strategy_perf": output["res"]["strategy_perf"].iloc[-1],
                "strategy_std": output["res"]["strategy_perf"].std(),
                "median_perf_delta": output["res"]["perf_delta"].median(),
            }

        if mean_metric_value > 0:
            json_data = json.dumps(seed_res, indent=4)
            save_path = PROJECT_ROOT / json_path
            mode = "a" if save_path.exists() else "w"
            with open(save_path, mode) as file:
                file.write(json_data)

        return mean_metric_value
