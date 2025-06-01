import pandas as pd
import plotly.graph_objs as go

from portfolio_constructor import ALL_BASE_COLS_DESCRIPTIONS
from portfolio_constructor.feature_generator import base_features_data_generator


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


def generate_base_features():
    (
        df,
        _,
        _,
    ) = base_features_data_generator(path="mcftrr.xlsx")

    print(df.info())
    features = {
        ALL_BASE_COLS_DESCRIPTIONS.get(col_name, col_name): df[col_name]
        for col_name in df.columns
    }

    derivatives_data = {}

    for name, data in features.items():
        df = pd.DataFrame({"value": data}, index=data.index)
        derivatives_data[name] = df

    return derivatives_data


# def generate_base_features():
#     features = {
#         "Цена на нефть": pd.read_excel(
#             "data/exog_data/preprocessed/brent.xlsx", index_col="date"
#         )["brent"],
#         "Цена на золото": pd.read_excel(
#             "data/exog_data/preprocessed/gold.xlsx", index_col="date"
#         )["gold"],
#         "Курс USD/RUB": pd.read_excel(
#             "data/exog_data/preprocessed/usd.xlsx", index_col="date"
#         )["usd"],
#         "Облигации 10ти летние": pd.read_excel(
#             "data/exog_data/preprocessed/bonds10y.xlsx", index_col="date"
#         )["bonds10y"],
#     }

#     derivatives_data = {}
#     for name, data in features.items():
#         df = pd.DataFrame({"value": data}, index=data.index)
#         derivatives_data[name] = df

#     print(df)

#     return derivatives_data


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
