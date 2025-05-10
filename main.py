import argparse
import os
import sys
from warnings import simplefilter

import numpy as np
import pandas as pd

from utils.feature_filter import feature_filter
from utils.feature_generator import (
    data_generator,
    replace_old_feature_names,
)

# from utils.sampler import SampleStrategy
from utils.model import StrategyModeller, read_logger


def main(parser):
    is_debug_mode = getattr(sys, "gettrace", lambda: None)() is not None

    if is_debug_mode:
        args = argparse.Namespace(features="random", filter=False, sampling=False)
    else:
        args = parser.parse_args()

    if "data.csv" in os.listdir("data"):
        data = pd.read_parquet("data/data.parquet")
    else:
        data = data_generator(path="mcftrr.xlsx")
        data.to_parquet("data/data.parquet")

    if args.features == "all":
        features = list(data.drop(["price", "ruonia_daily"], axis=1).columns)
    elif args.features == "random":
        features = list(data.drop(["price", "ruonia_daily"], axis=1).columns)
        features = list(np.random.choice(features, 20, replace=False))
    elif args.features == "last_best":
        trials = read_logger()
        trials["features"] = trials["features"].apply(lambda x: tuple(x))
        group_median_perf_delta = (
            trials.groupby(["run_date", "features"])["median_perf_delta"]
            .mean()
            .reset_index()
        )
        trials = pd.merge(
            trials,
            group_median_perf_delta,
            how="left",
            on=["run_date", "features"],
            suffixes=("", "_mean"),
        )
        max_median_perf_delta = trials.loc[
            trials.groupby("run_date")["median_perf_delta_mean"].idxmax()
        ].sort_values("run_date", ascending=False)
        # features = max_median_perf_delta['features'].iloc[0]
        features = max_median_perf_delta.loc[
            max_median_perf_delta["run_date"] == "2025-03-25", "features"
        ].item()
        features = tuple(map(lambda x: replace_old_feature_names(x), features))

    elif args.features == "feature_importance":
        if "feature_importance.xlsx" not in os.listdir("data"):
            raise Exception(
                '"features_importance.xlsx" file is missing in folder "data"'
            )
        else:
            feature_importance = pd.read_excel(
                "data/feature_importance.xlsx", parse_dates=True
            )
            top_feature_importance = (
                feature_importance.mean().iloc[4:-1].sort_values(ascending=False)
            )
            features = list(
                top_feature_importance.loc[
                    top_feature_importance > top_feature_importance.mean()
                ].index[:100]
            )

    if args.filter:
        features = feature_filter(data, var_threshold=0.001, corr_threshold=0.85)

    top_n = 10
    mutual_info = pd.read_excel(
        "data/cum_mut_info.xlsx", index_col=[0], parse_dates=True
    )
    feature_info = mutual_info.apply(lambda x: list(x.nlargest(top_n).index), axis=1)

    step = 20
    splitter_kwargs = dict(
        forecast_horizon=[i for i in range(1, step + 1)],
        step_length=step,
        initial_window=500,
        window_length=None,
        eval_obs=0,
        set_sample_weights=True,
    )
    position_rotator_kwargs = dict(freq=63, shift_days=0, mode=1)
    sample_weight_kwargs = dict(
        weight_params=dict(
            # rotation_event=dict(
            #     imptnt_obs_lag_reaction=5,
            #     imptnt_obs_w=100,
            # ),
            dramatic_return_event=dict(
                return_mult=10,
                return_thresh=0.02,
            ),
            # compare_mean_return=dict(
            #     obs_frac=0.03,
            #     sqrt_scale=False,
            # ),
        ),
        weight_smoothing=dict(win_type="gauss", window=3, win_type_param=1),
    )
    model_kwargs = dict(
        model_name="catboost",
        n_models=2,
        iterations=10,
        subsample=0.8,
        random_state=2,
        # verbose=0,
        # l2_leaf_reg=10 if args.features == 'all' else 3,
        # grow_policy='Depthwise',
        # min_data_in_leaf=10,
        # thread_count=-1,
    )

    dummy_features = ["normal_dummy", "dummy"]

    if not args.sampling:
        model = StrategyModeller(
            splitter_kwargs,
            sample_weight_kwargs,
            position_rotator_kwargs,
            model_kwargs,
            prob_to_weight=True,
            logging=True,
            feature_info=feature_info,
        )
        batches = model.get_batches(data, features)
        preds, train_info = model.get_predictions(batches)
        strat_data = data.loc[
            :, ["price", "price_return", "ruonia", "ruonia_daily"]
        ].copy()
        output = model.base_strategy_peformance(strat_data, preds, plot=True)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument(
        "--features",
        choices=["all", "important", "last_best", "random"],
        default="all",
        help='whether to use all features ("all") or features from features_importance.xlsx ("important")',
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="variance and correlation feature filtering",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="whether to sample strategies with different feature subsets or sample one strategy with all features",
    )
    simplefilter("ignore")
    main(parser)


# зафитить деревья для каждого признака отдельно -> выбрать топ N по метрике
