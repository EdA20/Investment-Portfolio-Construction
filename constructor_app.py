from fastapi import Depends, FastAPI, Form, Request, BackgroundTasks  # noqa
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from warnings import simplefilter
from pydantic import BaseModel  # noqa
import typing as t  # noqa

import uuid  # noqa
import time  # noqa


simplefilter("ignore")

from portfolio_constructor.charts import (  # noqa
    features_data,
    generate_feature_chart,
    generate_price_chart,
)

from portfolio_constructor.feature_generator import data_generator  # noqa
from portfolio_constructor.model import Dataset, Model, Strategy  # noqa

from portfolio_constructor import ALL_BASE_COLS_DESCRIPTIONS

app = FastAPI()
app.mount("/static", StaticFiles(directory="data/static"), name="static")
templates = Jinja2Templates(directory="data/templates")

user_data = {}
selected_features = {}
derivatives = {}

users_db = {"admin": "admin123"}
tasks: t.Dict[str, t.Dict] = {}


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login/")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if users_db.get(username) == password:
        price_chart = generate_price_chart()
        return templates.TemplateResponse(
            "feature_selection.html",
            {
                "request": request,
                "features": list(ALL_BASE_COLS_DESCRIPTIONS.values()),
                "price_chart": price_chart,
            },
        )
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Неверные данные"}
    )


@app.post("/select-features/")
async def process_features(request: Request, selected: list = Form(...)):
    request.session["selected_features"] = selected

    feature_charts = {feature: generate_feature_chart(feature) for feature in selected}

    return templates.TemplateResponse(
        "derivatives.html",
        {
            "request": request,
            "features": selected,
            "feature_charts": feature_charts,
        },
    )


@app.post("/select-risk-profile/")
async def process_risk_profile(request: Request, risk_profile: str = Form(...)):
    print(request.session["selected_features"])
    request.session["risk_profile"] = risk_profile
    return templates.TemplateResponse("training.html", {"request": request})


@app.post("/select-derivatives/")
async def process_derivatives(request: Request):
    form_data = await request.form()
    derivatives = {}

    for key in form_data.keys():
        derivatives[key] = form_data.getlist(key)

    request.session["derivatives"] = derivatives
    return templates.TemplateResponse("risk_profile.html", {"request": request})


@app.post("/train-model/")
async def train_model(request: Request):
    risk_profile = request.session.get("risk_profile", "medium")
    return templates.TemplateResponse(
        "training.html", {"request": request, "risk_profile": risk_profile}
    )


class TrainResponse(BaseModel):
    task_id: str


@app.post("/train", response_model=TrainResponse)
async def start_training(background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "progress": 0}

    # Запускаем фоновую задачу
    background_tasks.add_task(train_model, task_id)

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})


# def train_model(task_id: str):
#     """Длительная операция обучения (замена на реальную логику)"""
#     PROGRESS_CNT = 20
#     for i in range(1, PROGRESS_CNT + 1):
#         time.sleep(0.3)  # Имитация работы
#         tasks[task_id]["progress"] = i

#     # Сохраняем результат
#     result = {
#         "status": "completed",
#         "progress": PROGRESS_CNT,
#         "accuracy": 0.85,
#         "risk_profile": "medium",  # Пример данных
#     }
#     tasks[task_id] = result
#     return result


def train_model(task_id: str):
    tasks[task_id]["progress"] = 100

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
        "roc": dict(cols=["price_return"], windows=[10, 21, 63, 126, 252]),
        "custom_aroon": dict(col="price", windows=[126, 252, 504]),
        #     "distribution_oscillator": dict(
        #         cols=["price_return"], windows=[252, 378], roc_windows=[21, 63, 126, 252] # лучше не исп-ть, будет долго считать
        #     ),
    }

    # определяем какие базовые признаки мы хотим исп-ть, например такие
    stat_gen_cols = [
        "price",
        "ruonia",
        "imoex_pe",
        "gold_return",
        "long/short_physic_ratio",
    ]

    stat_gen_kwargs = dict(
        columns=stat_gen_cols,
        attrs=("mean", "std", "skew", "kurt"),
        windows=(5, 10, 21, 63, 126),
        ratio_windows=((5, 21), (10, 21), (10, 63), (21, 63), (21, 126), (63, 126)),
        smoothing_types=("simple", "exponential"),
    )

    data_generator_kwargs = dict(ta_methods=ta_methods, stat_gen_kwargs=stat_gen_kwargs)
    data = data_generator(path="mcftrr.xlsx", **data_generator_kwargs)

    step = 20
    splitter_kwargs = dict(
        forecast_horizon=[i for i in range(1, step + 1)],
        step_length=step,
        initial_window=500,
        window_length=None,
        eval_obs=0,
        set_sample_weights=True,
    )

    position_rotator_kwargs_1 = dict(
        markup_name="triple_barrier",
        markup_kwargs=dict(h=250, shift_days=63, vol_span=20, volatility_multiplier=2),
    )

    position_rotator_kwargs_2 = dict(markup_name="min_max", markup_kwargs=dict(freq=63))

    sample_weight_kwargs = dict(
        weight_params=dict(
            time_critical=dict(k=100, q=0.01, jump_coef=20, fading_factor=21)
        )
    )

    model_kwargs = dict(
        model_name="catboost",
        n_models=1,
        iterations=10,
        subsample=0.8,
        random_state=2,
    )
    dataset = Dataset(
        splitter_kwargs,
        sample_weight_kwargs,
        position_rotator_kwargs_1,
    )
    features = list(
        data.drop(["price", "ruonia_daily"], axis=1, errors="ignore").columns
    )
    batches = dataset.get_batches(data, features)

    model = Model(
        dataset,
        model_kwargs,
    )
    preds, train_info = model.get_predictions(batches)

    strategy = Strategy(
        model,
        prob_to_weight=True,  # не округляет вероятность
    )

    strat_data = data.loc[:, ["price", "price_return", "ruonia", "ruonia_daily"]].copy()
    plot_kwargs = dict(
        perf_plot=True,
        sliding_plot=True,
        save=False,  # сразу рисует динамику портфеля
    )
    output = strategy.base_strategy_peformance(strat_data, preds, plot_kwargs)

    print(type(output))

    # Сохраняем результат
    result = {
        "status": "completed",
        "progress": 100,
        "accuracy": 0.85,
        "risk_profile": "medium",  # Пример данных
    }
    tasks[task_id] = result


app.add_middleware(
    SessionMiddleware,
    secret_key="your-very-secret-key-1234",
    session_cookie="algo_session",
)
