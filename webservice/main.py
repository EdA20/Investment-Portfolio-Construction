from fastapi import Depends, FastAPI, Form, Request  # noqa
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from charts import (  # noqa  # noqa
    features_data,
    generate_feature_chart,
    generate_price_chart,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

user_data = {}
selected_features = {}
derivatives = {}

users_db = {"admin": "admin123"}


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
                "features": [
                    "Цена на нефть",
                    "Цена на золото",
                    "Курс USD/RUB",
                    "Облигации 10ти летние",
                ],
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


app.add_middleware(
    SessionMiddleware,
    secret_key="your-very-secret-key-1234",
    session_cookie="algo_session",
)
