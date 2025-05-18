import sys
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from tqdm import tqdm

from portfolio_constructor import ALL_DATA_FILES, ALL_DATA_PATH, PROJECT_ROOT
from portfolio_constructor.target_markup import position_rotator


def group_features(features_to_group: List[str], grouping_lvl: str = "base_cols"):
    features_to_group = features_to_group.copy()
    exog_data_name_path_dct = dict(zip(ALL_DATA_FILES, ALL_DATA_PATH))
    feature_groups = {}
    attrs = ["mean", "std"]

    for name, path in exog_data_name_path_dct.items():
        temp = pd.read_excel(path, index_col=[0])
        grouped_cols = []

        if grouping_lvl == "base_cols":
            for col in temp.columns:
                features = list(filter(lambda x: x.startswith(col), features_to_group))
                grouped_cols.extend(features)
            group_name = name
            feature_groups[group_name] = grouped_cols.copy()
            features_to_group = list(
                set(features_to_group) - set(feature_groups[group_name])
            )

        elif grouping_lvl == "base_cols_attrs":
            for attr in attrs:
                for col in temp.columns:
                    features = list(
                        filter(
                            lambda x: x.startswith(col) and attr in x, features_to_group
                        )
                    )
                    grouped_cols.extend(features)
                group_name = f"{name}_{attr}"
                feature_groups[group_name] = grouped_cols.copy()
                features_to_group = list(
                    set(features_to_group) - set(feature_groups[group_name])
                )
        else:
            raise Exception(
                'Incorrect grouping_lvl argument. Possible values = "base_cols", "base_cols_attrs"'
            )

    feature_groups["other"] = features_to_group

    return feature_groups


def corr_feature_filter(
    data, features: Union[Iterable, None] = None, corr_threshold: float = None
):
    if corr_threshold is None:
        corr_threshold = 0.75

    if features:
        data = data.loc[:, features].copy()
    else:
        data = data.drop(["price", "ruonia_daily"], axis=1)
        features = data.columns

    corr_matrix = data.corr().abs()
    upper_matrix = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    features_to_drop = [
        column
        for column in upper_matrix.columns
        if any(upper_matrix[column] > corr_threshold)
    ]
    features = list(set(features) - set(features_to_drop))

    return features


def var_feature_filter(
    data: pd.DataFrame,
    features: Union[Iterable, None] = None,
    var_threshold: float = None,
):
    if var_threshold is None:
        var_threshold = 0.001

    if features:
        data = data.loc[:, features].copy()
    else:
        data = data.drop(["price", "ruonia_daily"], axis=1)
        features = data.columns

    selector = VarianceThreshold(threshold=var_threshold)
    selector.fit(data)
    mask = selector.get_support()
    features = [f for f, m in zip(features, mask) if m]

    return features


def feature_filter(
    data: pd.DataFrame, features: Union[Iterable, None] = None, **kwargs
):
    var_threshold = kwargs.get("var_threshold")
    corr_threshold = kwargs.get("corr_threshold")

    features = var_feature_filter(data, features, var_threshold)
    features = corr_feature_filter(data, features, corr_threshold)

    return features


def cv_mutual_info(
    data: pd.DataFrame, features: Union[Iterable, None] = None, cv: int = 5
):
    idx = np.linspace(0, len(data), cv + 1).astype(int)[1:] - 1
    mis = {}
    for end in tqdm(idx):
        end_date = data.index[end]

        target = data.loc[:end_date, "price"].copy()
        position = position_rotator(target, freq=63, shift_days=0)
        position["action"] = (position["action"] == "buy").astype(int)

        target.loc[:] = np.nan
        target.loc[:] = position["action"].copy()
        target = target.ffill()

        sample = data.loc[:end_date, features].ffill().copy()
        sample = sample.fillna(sample.mean()).fillna(0)

        mi = mutual_info_classif(sample, target, n_neighbors=3)
        mis[end_date] = mi.copy()

    mis = pd.DataFrame(mis, index=features).T

    return mis


def main(arg_parser):
    is_debug_mode = getattr(sys, "gettrace", lambda: None)() is not None
    if is_debug_mode:
        args = argparse.Namespace(var=0.001, corr=0.85, folds=15)
    else:
        args = arg_parser.parse_args()

    data = pd.read_parquet(PROJECT_ROOT / "data/data.parquet")
    features = feature_filter(data, var_threshold=args.var, folds=args.corr)
    cum_mut_info = cv_mutual_info(data, features, cv=args.folds)

    cum_mut_info.to_excel(PROJECT_ROOT / "data/cum_mut_info.xlsx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="feature_filter_parser")
    parser.add_argument(
        "--var",
        default=0.001,
        type=float,
        help="choose lower bound of variance for feature selection",
    )
    parser.add_argument(
        "--corr",
        default=0.85,
        type=float,
        help="choose upper bound of correlation for feature selection",
    )
    parser.add_argument(
        "--folds",
        default=15,
        type=float,
        help="number of folds to calculate mutual information",
    )
    main(parser)
