import pandas as pd
import plotly.graph_objs as go


def generate_price_chart():
    df = pd.read_excel("data/mcftrr.xlsx", index_col="date")["price"]

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
    dates = pd.date_range(start="2023-01-01", periods=100)

    features = {
        "Цена на нефть": pd.read_excel("data/brent.xlsx", index_col="date")["brent"],
        "Цена на золото": pd.read_excel("data/gold.xlsx", index_col="date")["gold"],
        "Курс USD/RUB": pd.read_excel("data/usd.xlsx", index_col="date")["usd"],
        "Облигации 10ти летние": pd.read_excel("data/bonds10y.xlsx", index_col="date")[
            "bonds10y"
        ],
    }

    derivatives_data = {}
    for name, data in features.items():
        df = pd.DataFrame({"value": data}, index=data.index)
        df["EMA_10"] = df["value"].ewm(span=10).mean()
        df["SMA_20"] = df["value"].rolling(20).mean()
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

    indicators = {
        "EMA_10": {"data": df["EMA_10"], "color": "#FF5722", "dash": "dot"},
        "SMA_20": {"data": df["SMA_20"], "color": "#4CAF50", "dash": "dash"},
    }

    for name, params in indicators.items():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=params["data"],
                name=name,
                line=dict(color=params["color"], dash=params["dash"]),
                visible=False,
            )
        )

    buttons = []
    for idx, (name, _) in enumerate(indicators.items(), start=1):
        buttons.append(
            {
                "method": "restyle",
                "label": f"☑ {name}",
                "args": [
                    {
                        "visible": [True]
                        + [i == idx for i in range(1, len(indicators) + 1)]
                    },
                    {"title": f"{feature_name} - {name}"},
                ],
            }
        )
    buttons.append(
        {
            "method": "restyle",
            "label": "☑ Все",
            "args": [
                {"visible": [True] + [True] * len(indicators)},
                {"title": f"{feature_name} - Все индикаторы"},
            ],
        }
    )

    fig.update_layout(
        title=f"{feature_name}",
        showlegend=True,
        height=400,
    )

    return fig.to_html(full_html=False)
