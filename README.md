# 📈 Investment Portfolio Construction: ML-Driven Strategy to Outperform IMOEX 

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Packaging-Poetry-cyan.svg)](https://python-poetry.org/)
[![DVC](https://img.shields.io/badge/Data_Versioning-DVC-yellowgreen)](https://dvc.org/)

A diploma project building a machine learning-powered investment strategy to **consistently outperform** the Moscow Exchange Index (IMOEX). Combines equity selection with risk-managed portfolio construction using RUONIA rates.

---

## 🎯 Project Overview

#### Objective
Develop an **automated trading pipeline** using ML/DL models to generate buy/sell signals, constructing diversified portfolios that:
1. Outperform IMOEX returns
2. Mitigate risk through sector diversification and RUONIA rate hedging

#### Key Innovations
- **Hybrid strategy**: Combines equities with RUONIA deposits for downside protection
- **Multi-source data**: Integrates endogenous (price data) and exogenous (commodities, FX, bonds) features
- **Advanced feature engineering**: Technical indicators + macroeconomic derivatives

---

## 📊 Visualizations

#### Strategy performance

![alt text](https://github.com/EdA20/ImoexMLProject/blob/master/_images/strategy_performance.png?raw=true)

#### Strategy performance detailed

![alt text](https://github.com/EdA20/ImoexMLProject/blob/master/_images/strategy_perf_detailed.png?raw=true)

---

## 🏆 Key Results

...

---

## 🧠 Methodology

#### Data Pipeline
```mermaid
graph LR
A[Raw Data] --> B[Feature Engineering]
B --> C[Model Training]
C --> D[Signal Generation]
D --> E[Portfolio Construction]
E --> F[Performance Analysis]
```

---

## ⚙️ Installation & Usage

#### Requirements
- `python >=3.10<3.12`
- `poetry >= 2.1.3` (dependency management)
- `dvc` (data version control)

#### Data Management (DVC)

```bash
# Configure DVC
poetry run dvc remote modify s3-portfolio-construction --local access_key_id $DVC_ACCESS_KEY_ID
poetry run dvc remote modify s3-portfolio-construction --local secret_access_key $DVC_SECRET_ACCESS_KEY

# Pull/Push data
poetry run dvc pull
poetry run dvc push
```

#### Installation

```bash
poetry install
```


#### Running service locally

```bash
cd webservice
poetry run uvicorn main:app --reload
```

---

## 📂 Project Structure
...


## 🤝 Contribution Guidelines

#### Setup pre-commit hooks:

```bash
poetry run pre-commit install
```

#### Format code:

```bash
poetry run ruff format .
```

#### Add new dependencies:

```bash
poetry add <package> --group dev  # For development tools
poetry add <package> --group main # For core dependencies
```