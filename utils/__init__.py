from pathlib import Path

TQDM_DISABLE = False

ENDOG_DATA_FILE = ['mcftrr']

EXOG_DATA_RATE_FILES = ['bonds10y', 'ruonia']
EXOG_DATA_PRICE_FILES = ['brent', 'gold', 'usd', 'mredc']
EXOG_DATA_OTHER_FILES = ['top3_mean_vol', 'imoex_pe', 'mix_pos_struct']

ALL_EXOG_DATA_FILES = EXOG_DATA_RATE_FILES + EXOG_DATA_PRICE_FILES + EXOG_DATA_OTHER_FILES
ALL_DATA_FILES = ALL_EXOG_DATA_FILES + ENDOG_DATA_FILE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXOG_DATA_FOLDER = PROJECT_ROOT / 'data/exog_data/preprocessed/'
ENDOG_DATA_FOLDER = PROJECT_ROOT / 'data/endog_data/'

ALL_EXOG_DATA_PATH = [EXOG_DATA_FOLDER / (name + '.xlsx') for name in ALL_EXOG_DATA_FILES]
ENDOG_DATA_PATH = [ENDOG_DATA_FOLDER / (name + '.xlsx') for name in ENDOG_DATA_FILE]
ALL_DATA_PATH = ALL_EXOG_DATA_PATH + ENDOG_DATA_PATH
