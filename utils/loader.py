import requests
import pandas as pd

from tqdm import tqdm
from typing import Union, List
from utils import PROJECT_ROOT

RAW_FILES_FOLDER = PROJECT_ROOT / 'data/exog_data/raw/'
PREPROCESSED_FILES_FOLDER = PROJECT_ROOT / 'data/exog_data/preprocessed/'

CBR_BASE_URL = 'https://cbr.ru/hd_base'
MOEX_BASE_URL = "https://iss.moex.com"


def moex_history(
    engine: str = 'stock',
    market: str = 'shares',
    secids: Union[None, List[str]] = None,
    method: str = 'securities',
    response_format: str = 'json',
    from_date: str = '2020-12-01',
    till_date: str = None,
):
    if secids is None:
        secids = ['SBER']

    url = MOEX_BASE_URL + '/iss/history/'
    securities = []

    for i in tqdm(range(len(secids))):

        market_obj = {
            'engines': engine,
            'markets': market,
            'securities': secids[i]
        }

        query = ''
        for key, value in market_obj.items():
            query += f'{key}/{value}/'
        query += f'{method}.{response_format}'

        query_params = {
            'start': 0,
            'from': from_date,
            'till': till_date if till_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        if engine == 'stock' and market in ('shares', 'bonds', 'foreignshares'):
            query_params['marketprice_board'] = '1'

        resp = requests.get(url + query, params=query_params).json()
        index, total, page_size = resp['history.cursor']['data'][0]
        total = (total // page_size + 1) * page_size

        for start_index in tqdm(range(0, total + 1, page_size)):
            query_params['start'] = start_index
            response = requests.get(
                url + query, params=query_params, verify=False
            )
            asset_price = response.json()
            data = asset_price['history']['data']
            cols = list(
                map(lambda x: x.lower(), asset_price['history']['columns'])
            )
            df = pd.DataFrame(data, columns=cols)
            securities.append(df)

    securities = pd.concat(securities).sort_values('tradedate', ignore_index=True)
    securities['tradedate'] = pd.to_datetime(securities['tradedate'])

    return securities


def get_cbr_ruonia(till_date: str = None, file_name='ruonia.xlsx'):
    output_path = PREPROCESSED_FILES_FOLDER / file_name
    page_dir = '/ruonia/dynamics/'

    payload = {
        'UniDbQuery.Posted': True,
        'UniDbQuery.From': '01.01.2010',
        'UniDbQuery.To': till_date or pd.Timestamp.now().strftime('%d.%m.%Y')
    }
    url_params = []
    for key, value in payload.items():
        url_params.append(f'{key}={value}')
    url_params = '&'.join(url_params)

    url = f'{CBR_BASE_URL}{page_dir}?{url_params}'
    df = pd.read_html(url, decimal=',', thousands=' ')[0]
    df = df.iloc[:, :2].set_axis(['date', 'ruonia'], axis=1)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.set_index('date').sort_index()
    df.to_excel(output_path)

    return df


def get_cbr_gold_price(till_date: str = None, file_name='gold_rub.xlsx'):
    output_path = PREPROCESSED_FILES_FOLDER / file_name
    page_dir = '/metall/metall_base_new/'

    payload = {
        'UniDbQuery.Posted': True,
        'UniDbQuery.From': '01.01.2010',
        'UniDbQuery.To': till_date or pd.Timestamp.now().strftime('%d.%m.%Y'),
        'UniDbQuery.Gold': True,
        'UniDbQuery.so': 1
    }
    url_params = []
    for key, value in payload.items():
        url_params.append(f'{key}={value}')
    url_params = '&'.join(url_params)

    url = f'{CBR_BASE_URL}{page_dir}?{url_params}'
    df = pd.read_html(url, decimal=',', thousands=' ')[0]
    df = df.iloc[:, :2].set_axis(['date', 'gold'], axis=1)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.to_excel(output_path, index=False)

    return df


if __name__ == '__main__':
    engines = ['stock']
    market = ['shares', 'index']

    kwargs = dict(
        engine='stock',
        market='index',
        secids=['MCFTRR'],
        method='securities',
        response_format='json',
        from_date='2010-12-01',
        till_date=None,
    )
    df = moex_history(**kwargs)
    df = df[['tradedate', 'close']].drop_duplicates()
    df = df.set_axis(['date', 'price'], axis=1)
    df1 = pd.read_excel('data/endog_data/mcftrr.xlsx')

    df = pd.concat([df1, df]) \
        .drop_duplicates('date') \
        .sort_values('date') \
        .set_index('date')

    df.to_excel('data/endog_data/mcftrr.xlsx')
