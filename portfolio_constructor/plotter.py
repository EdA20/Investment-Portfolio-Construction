import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import clear_output
from plotly.subplots import make_subplots
from tqdm import tqdm

from portfolio_constructor import (
    EXOG_DATA_OTHER_FILES,
    EXOG_DATA_PRICE_FILES,
    PROJECT_ROOT,
)
from portfolio_constructor.target_markup import position_rotator

sns.set_style("darkgrid")


SAVE_PLOT_DIR = "plots"


def plot_position_rotator(price, min_max, freq):
    # Подготовка данных
    sell = min_max.loc[min_max["action"] == "sell", "value"]
    buy = min_max.loc[min_max["action"] == "buy", "value"]

    # Создание фигуры
    fig = go.Figure()

    # Линия цены
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name="Цена",
            line=dict(color="blue", width=2),
            hovertemplate="Дата: %{x}<br>Цена: %{y:.2f}<extra></extra>",
        )
    )

    # Точки продажи
    fig.add_trace(
        go.Scatter(
            x=sell.index,
            y=sell.values,
            mode="markers",
            name="Продажа",
            marker=dict(color="green", size=10, line=dict(width=1, color="DarkGreen")),
            visible=True,
            # hovertemplate="Продажа<br>Дата: %{x}<br>Цена: %{y:.2f}<extra></extra>",
        )
    )

    # Точки покупки
    fig.add_trace(
        go.Scatter(
            x=buy.index,
            y=buy.values,
            mode="markers",
            name="Покупка",
            marker=dict(color="red", size=10, line=dict(width=1, color="DarkRed")),
            visible=True,
            # hovertemplate="Покупка<br>Дата: %{x}<br>Цена: %{y:.2f}<extra></extra>",
        )
    )

    # Настройка layout
    fig.update_layout(
        title=f"Точки входа/выхода с частотой {freq}",
        xaxis_title="Дата",
        yaxis_title="Цена",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
        ),
        # hovermode="x unified",
        template="plotly_white",
        height=500,
        # width=1000,
        # margin=dict(l=50, r=50, b=50, t=80),
    )

    # fig.show()

    return fig.to_html(full_html=False)


# def plot_position_rotator(price, freqs, shift_days=0, mode=1):
#     fig, ax = plt.subplots(len(freqs), figsize=(10, 4 * len(freqs)), dpi=100)
#     if len(freqs) == 1:
#         ax = [ax]
#
#     for i, freq in enumerate(freqs):
#         min_max = position_rotator(price, freq, shift_days, mode)
#
#         sell = min_max.loc[min_max["action"] == "sell", "value"]
#         buy = min_max.loc[min_max["action"] == "buy", "value"]
#
#         ax[i].plot(price)
#         ax[i].scatter(sell.index, sell.values, s=40, c="g", label="sell")
#         ax[i].scatter(buy.index, buy.values, s=40, c="r", label="buy")
#         ax[i].set_title(f"Точки входа/выхода с частотой {freq}")
#         ax[i].legend(fontsize=13)
#
#     plt.tight_layout()
#     plt.show()


def plot_train_position_rotator(
    price, freqs, shift_days=0, mode=2, init_window=2000, step=21
):
    fig, ax = plt.subplots(len(freqs), figsize=(10, 4 * len(freqs)), dpi=100)
    if len(freqs) == 1:
        ax = [ax]

    for i, freq in enumerate(freqs):
        df_min_max = []
        for end in tqdm(np.arange(init_window, len(price), step)):
            min_max = position_rotator(
                price.iloc[:end], freq, shift_days, mode
            ).reset_index()
            df_min_max.append(min_max)
            clear_output(wait=True)

        min_max = pd.concat(df_min_max).drop_duplicates(subset="date").set_index("date")

        sell = min_max.loc[min_max["action"] == "sell", "value"]
        buy = min_max.loc[min_max["action"] == "buy", "value"]

        ax[i].plot(price)
        ax[i].scatter(sell.index, sell.values, s=40, c="g", label="sell")
        ax[i].scatter(buy.index, buy.values, s=40, c="r", label="buy")
        ax[i].set_title(f"Точки входа/выхода с частотой {freq}")
        ax[i].legend(fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_pos_rotator_min_max(price, freq, shift_days=0, mode=1):
    if isinstance(freq, str):
        mins = price.loc[price.groupby(pd.Grouper(level=0, freq=freq)).idxmin()]
        maxs = price.loc[price.groupby(pd.Grouper(level=0, freq=freq)).idxmax()]
    else:
        price = price.reset_index()
        price["index_group"] = price.index // (freq + 1)
        price = price.set_index("date")
        mins = price.loc[price.groupby("index_group")["price"].idxmin(), "price"]
        maxs = price.loc[price.groupby("index_group")["price"].idxmax(), "price"]
        price = price.drop("index_group", axis=1).squeeze()

    df_mins = pd.concat([mins.shift(), mins, mins.shift(-1)], axis=1).set_axis(
        ["min_f", "min", "min_b"], axis=1
    )
    df_maxs = pd.concat([maxs.shift(), maxs, maxs.shift(-1)], axis=1).set_axis(
        ["max_f", "max", "max_b"], axis=1
    )

    df_mins["global_min"] = df_mins.apply(
        lambda x: (x["min_f"] > x["min"]) & (x["min_b"] > x["min"]), axis=1
    )
    df_mins.iloc[0, -1] = True

    df_maxs["global_max"] = df_maxs.apply(
        lambda x: (x["max_f"] < x["max"]) & (x["max_b"] < x["max"]), axis=1
    )
    df_maxs.iloc[0, -1] = True

    df_maxs["downtrend"] = (df_maxs["max_f"] - df_maxs["max_b"]) / df_maxs["max_b"] > 0
    df_maxs["global_max_adj"] = df_maxs["global_max"] & df_maxs["downtrend"]
    last_max_adj_idx = df_maxs["global_max_adj"].cumsum().idxmax()
    last_max_idx = df_maxs["global_max"].cumsum().idxmax()
    df_maxs.loc[last_max_adj_idx + pd.Timedelta(days=1) :, "global_max"] = False

    sup_points_idx = [
        (df_mins.index[-2], df_mins.index[-3]),
        (df_mins.index[-2], df_mins.index[-4]),
        (df_mins.index[-3], df_mins.index[-4]),
    ]
    price_increments = []
    for right, left in sup_points_idx:
        price_increment = (df_mins.loc[right, "min"] - df_mins.loc[left, "min"]) / (
            right - left
        ).days
        price_increments.append(price_increment)
    mean_price_incr = np.mean(price_increments)

    if mean_price_incr > 0:
        delta_days = (price.index[-1] - df_mins.index[-2]).days
        sup_lvl = delta_days * mean_price_incr + df_mins.loc[df_mins.index[-2], "min"]
        cond = (price.iloc[-1] < sup_lvl) & (
            last_max_idx > df_mins["global_min"].cumsum().idxmax()
        )
        df_maxs.loc[last_max_idx, "global_max"] = True if cond else False

    global_dates = {
        "sell": df_maxs[df_maxs["global_max"]].index,
        "buy": df_mins[df_mins["global_min"]].index,
    }
    res = []
    for action, action_date_idx in global_dates.items():
        shifted_action_date_idx = (
            np.where(price.index.isin(action_date_idx))[0] + shift_days
        )
        action_ser = pd.Series([action for i in range(len(shifted_action_date_idx))])
        to_concat = [price.iloc[shifted_action_date_idx].reset_index(), action_ser]
        res.append(
            pd.concat(to_concat, axis=1)
            .set_index("date")
            .set_axis(["value", "action"], axis=1)
        )

    fig, ax = plt.subplots(2, figsize=(12, 8))
    ax[0].plot(price)
    if mean_price_incr > 0:
        ax[0].scatter(price.index[-1], sup_lvl, c="orange")
    ax[0].scatter(
        df_maxs["max"].index, df_maxs["max"].values, s=40, c="g", label="sell"
    )
    ax[0].scatter(df_mins["min"].index, df_mins["min"].values, s=40, c="r", label="buy")
    ax[0].set_title(f"Точки входа/выхода с частотой {freq}")
    ax[0].legend(fontsize=13)

    min_max = position_rotator(price, freq, shift_days, mode=mode)
    sell = min_max.loc[min_max["action"] == "sell", "value"]
    buy = min_max.loc[min_max["action"] == "buy", "value"]

    ax[1].plot(price)
    ax[1].scatter(sell.index, sell.values, s=40, c="g", label="sell")
    ax[1].scatter(buy.index, buy.values, s=40, c="r", label="buy")
    ax[1].set_title(f"Точки входа/выхода с частотой {freq}")
    ax[1].legend(fontsize=13)
    plt.show()


def plot_endog_vs_exog_index(data, return_cols=None, in_pycharm=True):
    if return_cols is None:
        all_return_cols = [
            col + "_return" for col in EXOG_DATA_PRICE_FILES + EXOG_DATA_OTHER_FILES
        ]
        return_cols = list(filter(lambda x: x in data.columns, all_return_cols))
    if in_pycharm:
        for i, col in enumerate(return_cols):
            plt.figure(figsize=(12, 8))
            (100 * (1 + data[col]).cumprod()).plot(label=col)
            (100 * (1 + data["return"]).cumprod()).plot(label="return")
            plt.legend()
    else:
        fig, ax = plt.subplots(len(return_cols), figsize=(8 * len(return_cols), 12))
        for i, col in enumerate(return_cols):
            (100 * (1 + data[col]).cumprod()).plot(label=col, ax=ax[i])
            (100 * (1 + data["return"]).cumprod()).plot(label="return", ax=ax[i])
            ax[i].legend()


def plot_sliding_pnl(res, days=365 * 2, save=True):
    # Проверка колонок
    cols = res.columns
    if "strat_return" not in cols:
        raise Exception('input dataframe must contain "strat_return" column')
    if "price_return" not in cols:
        raise Exception('input dataframe must contain "price_return" column')

    # Расчет скользящей доходности
    res[f"strat_hist_pnl_{days}d"] = (
        res["strat_return"]
        .rolling(f"{days}D")
        .apply(lambda x: 100 * (1 + x).prod() - 100)
    )
    res[f"moex_hist_pnl_{days}d"] = (
        res["price_return"]
        .rolling(f"{days}D")
        .apply(lambda x: 100 * (1 + x).prod() - 100)
    )

    # Создание графика Plotly
    fig = go.Figure()

    # Добавление линий
    fig.add_trace(
        go.Scatter(
            x=res.index,
            y=res[f"strat_hist_pnl_{days}d"],
            name="Portfolio Return",
            line=dict(color="blue", width=2),
            visible=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=res.index,
            y=res[f"moex_hist_pnl_{days}d"],
            name="MCFTRR Return",
            line=dict(color="red", width=2),
            visible=True,
        )
    )

    # Настройка layout
    fig.update_layout(
        title=f"Portfolio vs MCFTRR {days} days PnL",
        xaxis_title="Date",
        yaxis_title="PnL (%)",
        legend_title="Strategy",
        template="plotly_white",
        autosize=True,
        height=500,
        title_x=0.5,  # Центрирование заголовка
    )

    if save:
        os.makedirs(PROJECT_ROOT / SAVE_PLOT_DIR, exist_ok=True)
        name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        saving_path = os.path.join(
            PROJECT_ROOT,
            f"{SAVE_PLOT_DIR}/sliding_pnl_{name}.png",
        )
        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis=dict(tickfont=dict(size=12), tickangle=-45),
            legend=dict(font=dict(size=15), x=1.02, y=0.5),
        )
        fig.write_image(
            saving_path,
            width=800,  # Ширина в пикселях
            height=500,  # Высота в пикселях
            scale=3,
            engine="kaleido",
        )

    # fig.show()
    return saving_path


def plot_strategy_performance(strategy_res, save=True):
    cols = strategy_res.columns
    if "strategy_perf" not in cols:
        raise Exception('input dataframe must contain "strategy_perf" column')
    if "bench_perf" not in cols:
        raise Exception('input dataframe must contain "bench_perf" column')

    fig = go.Figure()

    # Добавляем стратегию
    fig.add_trace(
        go.Scatter(
            x=strategy_res.index,
            y=strategy_res["strategy_perf"],
            name="Portfolio Performance",
            line=dict(color="blue", width=2),
        )
    )

    # Добавляем бенчмарк
    fig.add_trace(
        go.Scatter(
            x=strategy_res.index,
            y=strategy_res["bench_perf"],
            name="MCFTRR Performance",
            line=dict(color="red", width=2),
        )
    )

    # Настраиваем layout
    fig.update_layout(
        title="Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Performance",
        legend_title="Legend",
        hovermode="x unified",
        autosize=True,
        width=800,
        height=500,
        template="plotly_white",
        showlegend=True,
    )

    # Добавляем сетку
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    if save:
        os.makedirs(PROJECT_ROOT / SAVE_PLOT_DIR, exist_ok=True)
        name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        saving_path = os.path.join(
            PROJECT_ROOT, f"{SAVE_PLOT_DIR}/performance_{name}.png"
        )
        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis=dict(tickfont=dict(size=12), tickangle=-45),
            legend=dict(font=dict(size=15), x=1.02, y=0.5),
        )
        fig.write_image(
            saving_path,
            width=800,  # Ширина в пикселях
            height=500,  # Высота в пикселях
            scale=3,
            engine="kaleido",
        )
        time.sleep(1)

    # fig.show()
    return saving_path


def plot_shifted_strategy_with_benchmark(data, random_start_returns, save=True):
    strats_perf = random_start_returns.copy()
    moex_perf = data.loc[strats_perf.index, "price"].pct_change()

    perfs = {"strats_perf": strats_perf, "moex_perf": moex_perf}
    for name, df in perfs.items():
        df.iloc[0] = 0
        df = (df + 1).cumprod() * 100
        perfs[name] = df

    perf_distribution = perfs["strats_perf"].iloc[-1].sort_values()

    quantiles = [0.05, 0.5, 0.95]
    compare_perfs = {"bench_perf": perfs["moex_perf"]}
    for q in quantiles:
        q_iloc = [int(len(perf_distribution) * q)]
        col_name = perf_distribution.iloc[q_iloc].index[0]
        compare_perfs[q] = perfs["strats_perf"][col_name]
    compare_perfs = pd.DataFrame(compare_perfs)

    fig = go.Figure()

    # Добавляем линии для квантилей
    for q in quantiles:
        fig.add_trace(
            go.Scatter(
                x=compare_perfs.index,
                y=compare_perfs[q],
                name=f"{q} quantile of performance distribution",
                line=dict(
                    dash="solid" if q == 0.5 else "dash", width=2.5 if q == 0.5 else 1.5
                ),
                opacity=1 if q == 0.5 else 0.7,
            )
        )

    # Добавляем линию бенчмарка
    fig.add_trace(
        go.Scatter(
            x=compare_perfs.index,
            y=compare_perfs["bench_perf"],
            name="benchmark performance",
            line=dict(color="black", width=2),
        )
    )

    # Настраиваем layout
    fig.update_layout(
        title="Strategy Performace Distribution",
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Performance",
        legend_font_size=12,
        hovermode="x unified",
        autosize=True,
    )

    if save:
        os.makedirs(PROJECT_ROOT / SAVE_PLOT_DIR, exist_ok=True)
        name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        fig.write_image(
            f"{SAVE_PLOT_DIR}/quantile_shifted_performance_{name}.png",
            width=1200,  # Ширина в пикселях
            height=800,  # Высота в пикселях
            scale=3,
            engine="kaleido",
        )
        time.sleep(1)

    fig.show()


def plot_bootstrap_features_performance(
    random_features_perf, outperf_info=True, save=True
):
    if outperf_info:
        # Создаем фигуру с двумя горизонтальными субплoтами
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.25)

        # Цветовая схема с хорошей различимостью
        color_scale = px.colors.sequential.Plasma

        # Добавляем оба графика
        for i, col in enumerate(["mean_mean_outperf", "std_mean_outperf"], 1):
            fig.add_trace(
                go.Scatter(
                    x=random_features_perf["std_strategy_perf"],
                    y=random_features_perf["mean_strategy_perf"],
                    mode="markers",
                    marker=dict(
                        color=random_features_perf[col],
                        colorscale=color_scale,
                        showscale=True,
                        colorbar=dict(
                            title=f"{col.split('_')[0].title()} Mean Outperformance<br>&nbsp;",
                            len=0.5,
                            y=0.5,
                            x=0.4
                            if i == 1
                            else 1.05,  # Размещаем цветовые бары справа от каждого графика
                            orientation="v",
                        ),
                        size=12,
                        opacity=0.9,
                        line=dict(width=1, color="black"),
                    ),
                    hovertext=random_features_perf[col].round(2),
                    hoverinfo="x+y+text",
                    name=col.replace("_", " ").title(),
                ),
                row=1,
                col=i,
            )

            # Обновляем оси
            fig.update_xaxes(title_text="Std Performance", row=1, col=i)
            fig.update_yaxes(title_text="Mean Performance", row=1, col=i)

        # Общие настройки
        fig.update_layout(
            title_text="Strategy exploration via seed and feature space sampling",
            title_x=0.5,
            height=850,
            width=1900,
            template="plotly_white",
            hovermode="closest",
            margin=dict(l=50, r=150, b=80, t=80, pad=10),
            showlegend=False,
        )
    else:
        # Упрощенная версия с одним графиком
        fig = px.scatter(
            random_features_perf,
            x="std_strategy_perf",
            y="mean_strategy_perf",
            title="Strategy exploration via seed and feature space sampling",
        )
        fig.update_traces(
            marker=dict(size=12, opacity=0.9, line=dict(width=1, color="black"))
        )
        fig.update_layout(
            autosize=True,
            title_x=0.5,
            template="plotly_white",
            xaxis_title="Std Performance",
            yaxis_title="Mean Performance",
        )

    # Улучшаем видимость маркеров
    fig.update_traces(marker=dict(opacity=0.9), selector=dict(mode="markers"))

    if save:
        os.makedirs(PROJECT_ROOT / SAVE_PLOT_DIR, exist_ok=True)
        name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        fig.write_image(
            f"{SAVE_PLOT_DIR}/bootstrap_features_performance_{name}.png",
            width=1500,  # Ширина в пикселях
            height=1000,  # Высота в пикселях
            scale=3,
            engine="kaleido",
        )

    # Настройка отображения в браузере
    fig.show(config={"responsive": True})


def plot_losses(train_losses, test_losses):
    fig, axs = plt.subplots(1, figsize=(12, 8))

    axs.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs.plot(range(1, len(test_losses) + 1), test_losses, label="val")
    axs.set_ylabel("loss")
    axs.set_xlabel("epoch")
    axs.legend()

    return fig


if __name__ == "__main__":
    # case 1
    from portfolio_constructor import PROJECT_ROOT

    data = pd.read_excel(
        PROJECT_ROOT / "data/endog_data/mcftrr.xlsx", index_col=[0], parse_dates=True
    )
    position_rotator_kwargs = dict(
        # markup_name='triple_barrier',
        # markup_kwargs=dict(
        #     h=250, shift_days=63, vol_span=20, volatility_multiplier=2
        # ),
        markup_name="min_max",
        markup_kwargs=dict(freq=63, rotator_type=1),
    )
    min_max = position_rotator(data, **position_rotator_kwargs)
    plot_position_rotator(data, min_max, 63)
    a = 1

    # # case 2
    # from portfolio_constructor.model import open_random_features_perf_file
    # df = open_random_features_perf_file()
    # plot_bootstrap_features_performance(df)
    # a=1
