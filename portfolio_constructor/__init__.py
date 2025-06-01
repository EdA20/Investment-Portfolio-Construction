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
    "price_vol_adj_return",
    "price_vol_adj",
    "bonds10y_ruonia_delta",
    "inv_imoex_pe_bonds10y_delta",
]

ALL_BASE_COLS_DESCRIPTIONS = {col_name: col_name for col_name in ALL_BASE_COLS}
# ALL_BASE_COLS_DESCRIPTIONS = {
#     "price",
#     "bonds10y",
#     "ruonia",
#     "brent",
#     "gold",
#     "usd",
#     "mredc",
#     "top3_mean_vol",
#     "imoex_pe",
#     "long_entity",
#     "short_entity",
#     "long_physic",
#     "short_physic",
#     "long/short_entity_ratio",
#     "long/short_physic_ratio",
#     "price_return",
#     "brent_return",
#     "gold_return",
#     "usd_return",
#     "mredc_return",
#     "top3_mean_vol_return",
#     "imoex_pe_return",
#     "long_entity_return",
#     "short_entity_return",
#     "long_physic_return",
#     "short_physic_return",
#     "long/short_entity_ratio_return",
#     "long/short_physic_ratio_return",
#     "price_vol_adj_return",
#     "price_vol_adj",
#     "bonds10y_ruonia_delta",
#     "inv_imoex_pe_bonds10y_delta",
# }
