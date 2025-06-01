import sys

import pandas as pd
import plotly.graph_objs as go

sys.path.append("../")


ALL_BASE_COLS = [
    "price",
    "bonds10y",
    "ruonia",
    "brent",
    "gold",
    "usd",
    "mredc",
    "top3_mean_vol",
    "imoex_pe",
    "long_entity",
    "short_entity",
    "long_physic",
    "short_physic",
    "long/short_entity_ratio",
    "long/short_physic_ratio",
    "price_return",
    "brent_return",
    "gold_return",
    "usd_return",
    "mredc_return",
    "top3_mean_vol_return",
    "imoex_pe_return",
    "long_entity_return",
    "short_entity_return",
    "long_physic_return",
    "short_physic_return",
    "long/short_entity_ratio_return",
    "long/short_physic_ratio_return",
    "price_vol_adj_return",
    "price_vol_adj",
    "bonds10y_ruonia_delta",
    "inv_imoex_pe_bonds10y_delta",
]


def generate_price_chart():
    df = pd.read_excel("data/endog_data/mcftrr.xlsx", index_col="date")["price"]

    fig = go.Figure(data=go.Scatter(x=df.index, y=df, mode="lines", name="Цена"))

    fig.update_layout(
        title="Исторические данные индекса MCFTRR",
        xaxis_title="Дата",
        yaxis_title="Цена",
        template="plotly_white",
        height=400,
    )

    return fig.to_html(full_html=False)


# def generate_base_features():
#     (
#         df,
#         _,
#         _,
#     ) = base_features_data_generator(path="mcftrr.xlsx")


def generate_base_features():
    features = {
        "Цена на нефть": pd.read_excel("data/exog_data/brent.xlsx", index_col="date")[
            "brent"
        ],
        "Цена на золото": pd.read_excel("data/exog_data/gold.xlsx", index_col="date")[
            "gold"
        ],
        "Курс USD/RUB": pd.read_excel("data/exog_data/usd.xlsx", index_col="date")[
            "usd"
        ],
        "Облигации 10ти летние": pd.read_excel(
            "data/exog_data/bonds10y.xlsx", index_col="date"
        )["bonds10y"],
    }

    derivatives_data = {}
    for name, data in features.items():
        df = pd.DataFrame({"value": data}, index=data.index)
        derivatives_data[name] = df

    print(df)

    return derivatives_data


# Глобальные данные
features_data = generate_base_features()


def generate_feature_chart(feature_name):
    df = features_data[feature_name]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["value"],
            name=feature_name,
            line=dict(color="#2196F3"),
            visible=True,
        )
    )

    fig.update_layout(
        title=f"{feature_name}",
        showlegend=True,
        height=400,
    )

    return fig.to_html(full_html=False)
