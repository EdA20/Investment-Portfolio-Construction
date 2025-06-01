import typing as t
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as sts
from IPython.display import clear_output
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from portfolio_constructor import (
    ALL_EXOG_DATA_FILES,
    ENDOG_DATA_FOLDER,
    EXOG_DATA_FOLDER,
    EXOG_DATA_OTHER_FILES,
    EXOG_DATA_PRICE_FILES,
    EXOG_DATA_RATE_FILES,
)

# from tsfresh import extract_features, select_features, extract_relevant_features
# from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame, impute

# pd.set_option('display.max_columns', None)


class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

        # cols = np.array(['open', 'low', 'high', 'close'])
        # check_cols = np.isin(cols, data.columns)
        # assert check_cols.all(), f'need to add cols: {cols[~check_cols]}'

    def rsi(self, col="close", windows=[10], lags=[], moving_average_type="simple"):
        df = pd.concat(
            [self.data[f"{col}"], self.data[f"{col}"].shift()], axis=1
        ).set_axis([f"{col}_curr", f"{col}_prev"], axis=1)
        df["U"] = df.apply(
            lambda x: x[f"{col}_curr"] - x[f"{col}_prev"]
            if x[f"{col}_curr"] > x[f"{col}_prev"]
            else 0,
            axis=1,
        )
        df["D"] = df.apply(
            lambda x: x[f"{col}_prev"] - x[f"{col}_curr"]
            if x[f"{col}_curr"] < x[f"{col}_prev"]
            else 0,
            axis=1,
        )
        data = []
        for window in windows:
            if moving_average_type == "simple":
                weightning_name = "sw"
                df[f"rsi_sw_{window}"] = (
                    100
                    * df["U"].rolling(window).mean()
                    / (df["U"].rolling(window).mean() + df["D"].rolling(window).mean())
                )
            elif moving_average_type == "exponential":
                weightning_name = "ew"
                df[f"rsi_ew_{window}"] = (
                    100
                    * df["U"]
                    .ewm(alpha=1 / window, min_periods=window, adjust=False)
                    .mean()
                    / (
                        df["U"]
                        .ewm(alpha=1 / window, min_periods=window, adjust=False)
                        .mean()
                        + df["D"]
                        .ewm(alpha=1 / window, min_periods=window, adjust=False)
                        .mean()
                    )
                )
            else:
                raise Exception(
                    f'moving average type can be only simple or exponential, "{moving_average_type}" was passed'
                )
            data.append(df[f"rsi_{weightning_name}_{window}"])

        if len(data) > 0:
            data = pd.concat(data, axis=1)
            data_with_lags = [data]
            for lag in lags:
                cols = data.columns + f"_lag_{lag}"
                data_with_lags.append(data.shift(lag).set_axis(cols, axis=1))
            data = pd.concat(data_with_lags, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def macd(
        self,
        col="close",
        short_windows=[12],
        long_windows=[26],
        signal_windows=[9],
        cols_to_use=[0, 1, 2],
    ):
        windows = zip(short_windows, long_windows, signal_windows)
        data = []
        for short_window, long_window, signal_window in windows:
            macd_line = (
                self.data[col]
                .ewm(alpha=1 / short_window, min_periods=short_window, adjust=False)
                .mean()
                - self.data[col]
                .ewm(alpha=1 / long_window, min_periods=long_window, adjust=False)
                .mean()
            )
            signal_line = macd_line.ewm(
                alpha=1 / signal_window, min_periods=signal_window, adjust=False
            ).mean()
            macd_delta = macd_line - signal_line
            cols = [
                f"macd_line_{short_window}_{long_window}",
                f"macd_signal_{signal_window}",
                f"macd_delta_{short_window}_{long_window}_{signal_window}",
            ]
            macd_info = pd.concat(
                [macd_line, signal_line, macd_delta], axis=1
            ).set_axis(cols, axis=1)
            data.append(macd_info.iloc[:, cols_to_use])

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def stochastic_oscillator(
        self,
        col="close",
        high_col="high",
        low_col="low",
        windows_k=[5],
        windows_n=[3],
        cols_to_use=[0, 1],
        moving_average_type="simple",
    ):
        windows = zip(windows_k, windows_n)
        data = []
        for window_k, window_n in windows:
            oscillator = (
                self.data[col] - self.data[low_col].rolling(window_k).min()
            ) / (
                self.data[high_col].rolling(window_k).max()
                - self.data[low_col].rolling(window_k).min()
            )
            oscillator -= 0.5
            if moving_average_type == "simple":
                weightning_name = "sw"
                smoothed_oscillator = oscillator.rolling(window_n).mean()
            elif moving_average_type == "exponential":
                weightning_name = "ew"
                smoothed_oscillator = oscillator.ewm(
                    alpha=1 / window_n, min_periods=window_n, adjust=False
                ).mean()
            else:
                raise Exception(
                    f'moving average type can be only simple or exponential, "{moving_average_type}" was passed'
                )

            cols = [
                f"oscillator_{weightning_name}_{window_k}",
                f"smoothed_oscillator_{weightning_name}_{window_n}",
            ]
            oscillators = pd.concat([oscillator, smoothed_oscillator], axis=1).set_axis(
                cols, axis=1
            )
            data.append(oscillators.iloc[:, cols_to_use])

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def mfi(self, windows=[10]):
        data = []
        for window in windows:
            typical_price = (
                self.data["close"] + self.data["low"] + self.data["high"]
            ) / 3
            positive, negative = (
                typical_price > typical_price.shift(),
                typical_price < typical_price.shift(),
            )
            money_flow = typical_price * self.data["value"]

            money_flow_positive = money_flow.copy()
            money_flow_positive.loc[~positive] = 0

            money_flow_negative = money_flow.copy()
            money_flow_negative.loc[~negative] = 0

            money_flow_index = (
                100
                * money_flow_positive.rolling(window).sum()
                / (
                    money_flow_positive.rolling(window).sum()
                    + money_flow_negative.rolling(window).sum()
                )
            )
            money_flow_index.name = f"money_flow_index_{window}"
            data.append(money_flow_index)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def atr(self, windows=[10], moving_average_type="simple"):
        data = []
        for window in windows:
            df = pd.concat(
                [self.data["close"].shift(), self.data["low"], self.data["high"]],
                axis=1,
            ).set_axis(["close_prev", "low", "high"], axis=1)
            true_range = df.apply(
                lambda x: max(
                    x["high"] - x["low"],
                    abs(x["high"] - x["close_prev"]),
                    abs(x["low"] - x["close_prev"]),
                ),
                axis=1,
            )

            if moving_average_type == "simple":
                average_true_range = true_range.rolling(window).mean()
            elif moving_average_type == "exponential":
                average_true_range = true_range.ewm(
                    alpha=1 / window, min_periods=window, adjust=False
                ).mean()
            else:
                raise Exception(
                    f'moving_average_type takes only args "simple" or "exponential", "{moving_average_type}" was passed'
                )

            average_true_range.name = f"average_true_range_{window}"
            data.append(average_true_range)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def disparity(
        self, short_windows=[12], long_windows=[26], moving_average_type="simple"
    ):
        windows = zip(short_windows, long_windows)
        data = []
        for short_window, long_window in windows:
            if moving_average_type == "simple":
                disparity = (
                    self.data["close"].rolling(short_window).mean()
                    / self.data["close"].rolling(long_window).mean()
                )
            elif moving_average_type == "exponential":
                disparity = (
                    self.data["close"]
                    .ewm(alpha=1 / short_window, min_periods=short_window, adjust=False)
                    .mean()
                    / self.data["close"]
                    .ewm(alpha=1 / long_window, min_periods=long_window, adjust=False)
                    .mean()
                )
            else:
                raise Exception(
                    f'moving_average_type takes only args "simple" or "exponential", "{moving_average_type}" was passed'
                )
            disparity.name = f"disparity_{short_window}_{long_window}"
            data.append(disparity)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def custom_aroon(self, col="close", windows=[14], cols_to_use=[0, 1, 2]):
        data = []
        for window in windows:
            aroon_up = (
                self.data[col].rolling(window).apply(lambda x: 1 - x.argmax() / window)
            )
            aroon_down = (
                self.data[col].rolling(window).apply(lambda x: 1 - x.argmin() / window)
            )
            aroon_delta = aroon_down - aroon_up

            cols = [
                f"aroon_up_{window}",
                f"aroon_down_{window}",
                f"aroon_delta_{window}",
            ]
            aroon = pd.concat([aroon_up, aroon_down, aroon_delta], axis=1).set_axis(
                cols, axis=1
            )
            data.append(aroon.iloc[:, cols_to_use])

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def stat_to_price_ratios(self, col="close", stats=["max"], windows=[5]):
        data = []
        for window in windows:
            for stat in stats:
                rolling_stat_func = getattr(self.data[col].rolling(window), stat)
                chg = rolling_stat_func() / self.data[col]
                chg.name = f"{stat}/{col}_{window}"
                data.append(chg)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def roc(self, cols=["return"], windows=[10, 21, 63]):
        data = []
        for window in windows:
            roc = self.data[cols].rolling(window).apply(lambda x: (1 + x).prod() - 1)
            roc_cols = [f"{col}_roc_{window}" for col in cols]
            roc = roc.set_axis(roc_cols, axis=1)
            data.append(roc)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def bollinger_bands(
        self, col="close", n_sigmas=3, windows=[6, 12, 18], cols_to_use=[0, 1, 2, 3]
    ):
        data = []
        for window in windows:
            mean = self.data[col].rolling(window).mean()
            std = self.data[col].rolling(window).std()

            lower_band, upper_band = mean - n_sigmas * std, mean + n_sigmas * std
            buy_ind, sell_ind = self.data[col] < lower_band, self.data[col] > upper_band
            cols = [
                f"lower_band_{window}",
                f"upper_band_{window}",
                f"bb_buy_ind_{window}",
                f"bb_sell_ind_{window}",
            ]
            bollinger_bands = pd.concat(
                [lower_band, upper_band, buy_ind, sell_ind], axis=1
            ).set_axis(cols, axis=1)
            data.append(bollinger_bands.iloc[:, cols_to_use])

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    @staticmethod
    def _get_cdf(ser, col):
        if "return" in col:
            args = sts.t.fit(ser)
            dist = sts.t(*args)
        else:
            args = sts.lognorm.fit(ser)
            dist = sts.lognorm(*args)
        prob_shift = dist.cdf(ser.iloc[-1])

        return prob_shift

    def distribution_oscillator(
        self, cols=["return"], windows=[252, 252 * 1.5], roc_windows=[61, 121]
    ):
        data = []
        for roc_window in roc_windows:
            roc = (
                self.data[cols].rolling(roc_window).apply(lambda x: (1 + x).prod() - 1)
            )
            for col in cols:
                for window in windows:
                    tqdm.pandas()
                    dist_osc = (
                        roc[col]
                        .rolling(window)
                        .progress_apply(lambda x: self._get_cdf(x, col))
                    )
                    # dist_osc = dist_osc.apply(lambda x: x if abs(x) > 0.4 else 0)
                    dist_osc.name = f"prob_{window}_roc_{roc_window}_{col}"
                    data.append(dist_osc)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data

    def vwap(self, price_col="close", volume_col="volume", windows=[61, 121]):
        data = []
        for window in windows:
            price_vol = self.data[price_col] * self.data[volume_col]
            vwap = (
                price_vol.rolling(window).sum()
                / self.data[volume_col].rolling(window).sum()
            )
            vwap.name = f"vwap_{window}"
            data.append(vwap)

        if len(data) > 0:
            data = pd.concat(data, axis=1)
        else:
            data = pd.DataFrame([])

        return data


def rolling_statistic_generator(
    data,
    columns=("close",),
    attrs=("mean", "std"),
    windows=(21, 61, 121),
    ratio_windows=((21, 61), (21, 121)),
    smoothing_types=("simple",),
):
    stats = []
    del_cols = list(
        filter(lambda x: ("price" in x) and ("return" not in x), columns)
    ) + list(filter(lambda x: x in columns, EXOG_DATA_PRICE_FILES))
    drop_cols = []

    simple_attrs = ["min", "median", "max", "quantile", "skew", "kurt"]
    for w in windows:
        for attr in attrs:
            for smoothing_type in smoothing_types:
                if (attr in simple_attrs) or (smoothing_type == "simple"):
                    cols = [i + f"_sw_{attr}_{w}" for i in columns]
                    drop_cols.extend([i + f"_sw_mean_{w}" for i in del_cols])
                    func = getattr(data.loc[:, columns].rolling(w), attr)
                else:
                    cols = [i + f"_ew_{attr}_{w}" for i in columns]
                    drop_cols.extend([i + f"_ew_mean_{w}" for i in del_cols])
                    func = getattr(
                        data.loc[:, columns].ewm(span=w, min_periods=w), attr
                    )

                stat = func().set_axis(cols, axis=1)
                stats.append(stat)

    if len(stats) > 0:
        stats = pd.concat(stats, axis=1)
        stats = stats.loc[:, ~stats.columns.duplicated()].copy()
    else:
        stats = pd.DataFrame(stats)

    stats_compare = []
    if ratio_windows:
        for short_w, long_w in ratio_windows:
            for attr in attrs:
                for smoothing_type in smoothing_types:
                    if (attr in simple_attrs) or (smoothing_type == "simple"):
                        short_w_cols = [i + f"_sw_{attr}_{short_w}" for i in columns]
                        long_w_cols = [i + f"_sw_{attr}_{long_w}" for i in columns]
                        new_cols_div = [
                            i + f"_sw_{attr}_{short_w}/{long_w}" for i in columns
                        ]
                        new_cols_subs = [
                            i + f"_sw_{attr}_{short_w}_{long_w}" for i in columns
                        ]
                    else:
                        short_w_cols = [i + f"_ew_{attr}_{short_w}" for i in columns]
                        long_w_cols = [i + f"_ew_{attr}_{long_w}" for i in columns]
                        new_cols_div = [
                            i + f"_ew_{attr}_{short_w}/{long_w}" for i in columns
                        ]
                        new_cols_subs = [
                            i + f"_ew_{attr}_{short_w}_{long_w}" for i in columns
                        ]

                    div = pd.DataFrame(
                        stats[short_w_cols].values / stats[long_w_cols].values,
                        columns=new_cols_div,
                        index=stats.index,
                    )
                    subs = pd.DataFrame(
                        stats[short_w_cols].values - stats[long_w_cols].values,
                        columns=new_cols_subs,
                        index=stats.index,
                    )
                    stats_compare.append(div)
                    stats_compare.append(subs)

    if len(stats_compare) > 0:
        stats_compare = pd.concat(stats_compare, axis=1)
        stats_compare = stats_compare.loc[:, ~stats_compare.columns.duplicated()].copy()
    else:
        stats_compare = pd.DataFrame(stats_compare)

    stats = pd.concat([stats, stats_compare], axis=1)
    stats = stats.drop(drop_cols, axis=1)

    return stats


def time_features_generator(data):
    data["year"] = data.index.year
    data["month"] = data.index.month

    ohe = OneHotEncoder()
    time_features = ohe.fit_transform(data[["year", "month"]])
    cols = ohe.get_feature_names(["year", "month"])
    time_features = pd.DataFrame(
        time_features.toarray(), index=data.index, columns=cols
    )

    return time_features


def feature_construction(data, ta_methods, stat_gen_kwargs):
    ti = TechnicalIndicators(data)
    ta_data = [
        getattr(ti, method)(**ta_method_kwargs)
        for method, ta_method_kwargs in ta_methods.items()
    ]
    ta_data = pd.concat(ta_data, axis=1)
    stat_gen_data = rolling_statistic_generator(data, **stat_gen_kwargs)
    all_data = pd.concat([data.shift(), ta_data.shift(), stat_gen_data.shift()], axis=1)
    all_data["price"] = data["price"].copy()
    all_data["ruonia_daily"] = all_data["ruonia_daily"].bfill()

    return all_data


def base_features_data_generator(
    path: str = None,
    exog_data_file_names: t.List[str] = None,
) -> t.Tuple[pd.DataFrame, t.List[str], t.List[str]]:
    if exog_data_file_names is None:
        exog_data_file_names = ALL_EXOG_DATA_FILES
    exog_data_file_paths = [
        EXOG_DATA_FOLDER / (name + ".xlsx") for name in exog_data_file_names
    ]
    exog_data_name_path_dct = dict(zip(exog_data_file_names, exog_data_file_paths))

    instrument = pd.read_excel(ENDOG_DATA_FOLDER / path, index_col=[0])

    exog_data = []
    exog_data_cols = {}
    for name, path in exog_data_name_path_dct.items():
        temp = pd.read_excel(path, index_col=[0])
        exog_data_cols[name] = list(temp.columns)
        exog_data.append(temp)
    exog_data = pd.concat(exog_data, axis=1)

    # код написан с расчетом на, что в exog_data_file_names захочется передавать не все файлы
    # доп обработка файлов с процентными ставками
    rate_files_in = list(
        filter(lambda x: x in exog_data_cols.keys(), EXOG_DATA_RATE_FILES)
    )
    for name in rate_files_in:
        rate_cols = exog_data_cols[name]
        # перевожу ставку из процентов в доли, если она не в долях
        for rate_col in rate_cols:
            if exog_data[rate_col].mean() > 1:
                exog_data[rate_col] /= 100

    df = pd.merge(instrument, exog_data, "left", left_index=True, right_index=True)
    df = df.interpolate(
        method="linear", limit_direction="forward", limit_area="inside", axis=0
    )

    df["price_return"] = df["price"].pct_change()

    # код написан с расчетом на, что в exog_data_file_names захочется передавать не все файлы
    # подсчет процентного изменения ценовых (и не только) временных рядов
    calc_return_files_in = list(
        filter(
            lambda x: x in exog_data_cols.keys(),
            EXOG_DATA_PRICE_FILES + EXOG_DATA_OTHER_FILES,
        )
    )
    for name in calc_return_files_in:
        calc_return_cols = exog_data_cols[name]
        for col in calc_return_cols:
            df[f"{col}_return"] = df[col].pct_change()

    df["price_vol_adj_return"] = df["price_return"] * (1 + df["top3_mean_vol_return"])
    df["price_vol_adj"] = (
        df["price"].iloc[0] * (1 + df["price_vol_adj_return"]).cumprod()
    )
    df["bonds10y_ruonia_delta"] = df["bonds10y"] - df["ruonia"]
    df["inv_imoex_pe_bonds10y_delta"] = 1 / df["imoex_pe"] - df["bonds10y"]

    delta_days = np.insert((df.index[1:] - df.index[:-1]).days.astype(int), -1, 1)
    days_in_year = df.index.is_leap_year + 365
    df["ruonia_daily"] = (df["ruonia"] - 0.005) * delta_days / days_in_year

    return_cols = list(filter(lambda x: "return" in x, df.columns))

    return df, exog_data_cols, return_cols


def data_generator(
    path: str = None,
    exog_data_file_names: List[str] = None,
    ta_methods: dict = None,
    stat_gen_kwargs: dict = None,
):
    df, exog_data_cols, return_cols = base_features_data_generator(
        path=path, exog_data_file_names=exog_data_file_names
    )

    if not ta_methods:
        ta_methods = {
            "rsi": dict(
                col="price", windows=[10, 21, 63, 126], moving_average_type="simple"
            ),
            "stochastic_oscillator": dict(
                col="price",
                high_col="price",
                low_col="price",
                windows_k=[10, 21, 63, 126],
                windows_n=[5, 10, 21, 21],
                moving_average_type="simple",
                cols_to_use=[0],
            ),
            "stat_to_price_ratios": dict(
                col="price",
                stats=["mean", "min", "median", "max"],
                windows=[10, 21, 63, 126, 252],
            ),
            "roc": dict(cols=return_cols, windows=[10, 21, 63, 126, 252]),
            "custom_aroon": dict(col="price", windows=[126, 252, 504]),
            "distribution_oscillator": dict(
                cols=["price_return"],
                windows=[252, 378],
                roc_windows=[21, 63, 126, 252],
            ),
        }

    if not stat_gen_kwargs:
        stat_gen_cols = list(df.drop("ruonia_daily", axis=1).columns)
        stat_gen_kwargs = dict(
            columns=stat_gen_cols,
            attrs=("mean", "std", "skew", "kurt"),
            windows=(5, 10, 21, 63, 126),
            ratio_windows=((5, 21), (10, 21), (10, 63), (21, 63), (21, 126), (63, 126)),
            smoothing_types=("simple", "exponential"),
        )

    data = feature_construction(df, ta_methods, stat_gen_kwargs)
    clear_output(wait=True)

    not_stationary_cols = list(
        filter(lambda x: x in exog_data_cols.keys(), EXOG_DATA_PRICE_FILES)
    ) + ["price_vol_adj"]
    data = data.drop(not_stationary_cols, axis=1)

    return data


def replace_old_feature_names(x):
    dct = {
        "close": "price",
        "bonds": "bonds10y",
        "top3_vol": "top3_mean_vol",
        "p/e": "imoex_pe",
        "rates_delta": "bonds10y_ruonia_delta",
    }
    for old_value, new_value in dct.items():
        if old_value in x:
            x = x.replace(old_value, new_value)
            break

    return x


if __name__ == "__main__":
    # generate feature space
    df = data_generator(path="mcftrr.xlsx")
    a = 1
