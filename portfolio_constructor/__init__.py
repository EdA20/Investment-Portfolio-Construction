from pathlib import Path

TQDM_DISABLE = False

ENDOG_DATA_FILE = ["mcftrr"]

EXOG_DATA_RATE_FILES = ["bonds10y", "ruonia"]
EXOG_DATA_PRICE_FILES = ["brent", "gold", "usd", "mredc"]
EXOG_DATA_OTHER_FILES = ["top3_mean_vol", "imoex_pe", "mix_pos_struct"]

ALL_EXOG_DATA_FILES = (
    EXOG_DATA_RATE_FILES + EXOG_DATA_PRICE_FILES + EXOG_DATA_OTHER_FILES
)
ALL_DATA_FILES = ALL_EXOG_DATA_FILES + ENDOG_DATA_FILE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXOG_DATA_FOLDER = PROJECT_ROOT / "data/exog_data/preprocessed/"
ENDOG_DATA_FOLDER = PROJECT_ROOT / "data/endog_data/"

ALL_EXOG_DATA_PATH = [
    EXOG_DATA_FOLDER / (name + ".xlsx") for name in ALL_EXOG_DATA_FILES
]
ENDOG_DATA_PATH = [ENDOG_DATA_FOLDER / (name + ".xlsx") for name in ENDOG_DATA_FILE]
ALL_DATA_PATH = ALL_EXOG_DATA_PATH + ENDOG_DATA_PATH

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
    "price_vol_adj",
    "price_vol_adj_return",
    "bonds10y_ruonia_delta",
    "inv_imoex_pe_bonds10y_delta",
]

# ALL_BASE_COLS_DESCRIPTIONS = {col_name: col_name for col_name in ALL_BASE_COLS}
ALL_BASE_COLS_DESCRIPTIONS = {
    "price": "Индекс MCFTRR",
    "bonds10y": "Доходности к погашению 10ти летних ОФЗ",
    "ruonia": "Значение ставки RUONIA",
    "brent": "Цена на нефть Brent",
    "gold": "Цена на золото (ЦБ РФ)",
    "usd": "Курс USD/RUB (ЦБ РФ)",
    "mredc": "Индекс недвижимости MREDC",
    "top3_mean_vol": "Среднедневной объем торгов Сбера, Газпрома, Лукойла",
    "imoex_pe": "Индикатор P/E акций Мосбиржи",
    "long_entity": "Открытые Long позиции по MIX фьючерсу юр-лица",
    "short_entity": "Открытые Short позиции по MIX фьючерсу юр-лица",
    "long_physic": "Открытые Long позиции по MIX фьючерсу физ-лица",
    "short_physic": "Открытые Short позиции по MIX фьючерсу физ-лица",
    "long/short_entity_ratio": "Отношение Long/Short позиций по MIX фьючерсу юр-лиц",
    "long/short_physic_ratio": "Отношение Long/Short позиций по MIX фьючерсу физ-лиц",
    "price_return": "Дневное изменение индекса MCFTRR",
    "brent_return": "Дневное изменение цены на нефть Brent",
    "gold_return": "Дневное изменение цены на золото",
    "usd_return": "Дневное изменение курса USD/RUB",
    "mredc_return": "Дневное изменение индекса MREDC",
    "top3_mean_vol_return": "Дневное изменение среднедневного объема торгов Сбер/Газпром/Лукойл",
    "imoex_pe_return": "Дневное изменение индикатора P/E",
    "long_entity_return": "Дневное изменение Long позиций юр-лиц",
    "short_entity_return": "Дневное изменение Short позиций юр-лиц",
    "long_physic_return": "Дневное изменение Long позиций физ-лиц",
    "short_physic_return": "Дневное изменение Short позиций физ-лиц",
    "long/short_entity_ratio_return": "Дневное изменение отношения Long/Short позиций юр-лиц",
    "long/short_physic_ratio_return": "Дневное изменение отношения Long/Short позиций физ-лиц",
    "price_vol_adj": "Динамика индекса MCFTRR с поправкой на объем торгов Сбер/Газпром/Лукойл",
    "price_vol_adj_return": "Дневное изменений индекса MCFTRR с поправкой на объем торгов Сбер/Газпром/Лукойл",
    "bonds10y_ruonia_delta": "Разница между ставкой RUONIA и к погашению 10ти летних ОФЗ",
    "inv_imoex_pe_bonds10y_delta": "Риск премия фондового рынка по отношению к долговому",
}

REVERTED_ALL_BASE_COLS_DESCRIPTIONS = {
    col_desc: col_name for col_name, col_desc in ALL_BASE_COLS_DESCRIPTIONS.items()
}
