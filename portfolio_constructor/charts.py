import pandas as pd
import plotly.graph_objs as go

from portfolio_constructor import (
    ALL_BASE_COLS_DESCRIPTIONS,
    REVERTED_ALL_BASE_COLS_DESCRIPTIONS,
)
from portfolio_constructor.feature_generator import base_features_data_generator


def generate_price_chart():
    df = pd.read_excel("data/endog_data/mcftrr.xlsx", index_col="date")["price"]

    fig = go.Figure(
        data=go.Scatter(
            x=df.index, y=df, mode="lines", name="Цена", line={"color": "blue"}
        )
    )

    fig.update_layout(
        title="Исторические данные индекса MCFTRR",
        xaxis_title="Дата",
        yaxis_title="Цена",
        template="plotly_white",
        height=500,
    )

    return fig.to_html(full_html=False)


def generate_base_features():
    (
        df,
        _,
        _,
    ) = base_features_data_generator(path="mcftrr.xlsx")
    features = {
        ALL_BASE_COLS_DESCRIPTIONS.get(col_name, col_name): df[col_name]
        for col_name in df.columns
    }

    derivatives_data = {}

    for name, data in features.items():
        df = pd.DataFrame({"value": data}, index=data.index)
        derivatives_data[name] = df

    return derivatives_data


features_data = generate_base_features()


def generate_feature_chart(feature_name):
    df = features_data[feature_name]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["value"],
            name=REVERTED_ALL_BASE_COLS_DESCRIPTIONS.get(feature_name, feature_name),
            line={"color": "blue"},
            visible=True,
        )
    )

    fig.update_layout(
        title=f"{feature_name}",
        xaxis_title="Дата",
        yaxis_title="Значение",
        template="plotly_white",
        showlegend=True,
        height=500,
    )

    result = fig.to_html(full_html=False)
    return result
