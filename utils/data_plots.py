from utils import EXOG_DATA_PRICE_FILES, EXOG_DATA_OTHER_FILES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML, clear_output
from tqdm import tqdm
from feature_generator import position_rotator

sns.set_style('darkgrid')


def plot_position_rotator(price, freqs, shift_days=0, mode=1):
    fig, ax = plt.subplots(len(freqs), figsize=(10, 4*len(freqs)), dpi=100)
    if len(freqs) == 1:
        ax = [ax]

    for i, freq in enumerate(freqs):
        min_max = position_rotator(price, freq, shift_days, mode)

        sell = min_max.loc[min_max['action'] == 'sell', 'value']
        buy = min_max.loc[min_max['action'] == 'buy', 'value']

        ax[i].plot(price)
        ax[i].scatter(sell.index, sell.values, s=40, c='g', label='sell')
        ax[i].scatter(buy.index, buy.values, s=40, c='r', label='buy')
        ax[i].set_title(f'Точки входа/выхода с частотой {freq}')
        ax[i].legend(fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_train_position_rotator(price, freqs, shift_days=0, mode=2, init_window=2000, step=21):
    fig, ax = plt.subplots(len(freqs), figsize=(10, 4*len(freqs)), dpi=100)
    if len(freqs) == 1:
        ax = [ax]

    for i, freq in enumerate(freqs):

        df_min_max = []
        for end in tqdm(np.arange(init_window, len(price), step)):
            min_max = position_rotator(price.iloc[:end], freq, shift_days, mode).reset_index()
            df_min_max.append(min_max)
            clear_output(wait=True)

        min_max = pd.concat(df_min_max).drop_duplicates(subset='date').set_index('date')

        sell = min_max.loc[min_max['action'] == 'sell', 'value']
        buy = min_max.loc[min_max['action'] == 'buy', 'value']

        ax[i].plot(price)
        ax[i].scatter(sell.index, sell.values, s=40, c='g', label='sell')
        ax[i].scatter(buy.index, buy.values, s=40, c='r', label='buy')
        ax[i].set_title(f'Точки входа/выхода с частотой {freq}')
        ax[i].legend(fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_pos_rotator_min_max(price, freq, shift_days=0, mode=1):
    if isinstance(freq, str):
        mins = price.loc[price.groupby(pd.Grouper(level=0, freq=freq)).idxmin()]
        maxs = price.loc[price.groupby(pd.Grouper(level=0, freq=freq)).idxmax()]
    else:
        price = price.reset_index()
        price['index_group'] = price.index // (freq + 1)
        price = price.set_index('date')
        mins = price.loc[price.groupby('index_group')['price'].idxmin(), 'price']
        maxs = price.loc[price.groupby('index_group')['price'].idxmax(), 'price']
        price = price.drop('index_group', axis=1).squeeze()

    df_mins = pd.concat([mins.shift(), mins, mins.shift(-1)], axis=1).set_axis(['min_f', 'min', 'min_b'], axis=1)
    df_maxs = pd.concat([maxs.shift(), maxs, maxs.shift(-1)], axis=1).set_axis(['max_f', 'max', 'max_b'], axis=1)

    df_mins['global_min'] = df_mins.apply(lambda x: (x['min_f'] > x['min']) & (x['min_b'] > x['min']), axis=1)
    df_mins.iloc[0, -1] = True

    df_maxs['global_max'] = df_maxs.apply(lambda x: (x['max_f'] < x['max']) & (x['max_b'] < x['max']), axis=1)
    df_maxs.iloc[0, -1] = True

    df_maxs['downtrend'] = (df_maxs['max_f'] - df_maxs['max_b']) / df_maxs['max_b'] > 0
    df_maxs['global_max_adj'] = df_maxs['global_max'] & df_maxs['downtrend']
    last_max_adj_idx = df_maxs['global_max_adj'].cumsum().idxmax()
    last_max_idx = df_maxs['global_max'].cumsum().idxmax()
    df_maxs.loc[last_max_adj_idx + pd.Timedelta(days=1):, 'global_max'] = False

    sup_points_idx = [
        (df_mins.index[-2], df_mins.index[-3]),
        (df_mins.index[-2], df_mins.index[-4]),
        (df_mins.index[-3], df_mins.index[-4])
    ]
    price_increments = []
    for right, left in sup_points_idx:
        price_increment = (df_mins.loc[right, 'min'] - df_mins.loc[left, 'min']) / (right - left).days
        price_increments.append(price_increment)
    mean_price_incr = np.mean(price_increments)

    if mean_price_incr > 0:
        delta_days = (price.index[-1] - df_mins.index[-2]).days
        sup_lvl = delta_days * mean_price_incr + df_mins.loc[df_mins.index[-2], 'min']
        cond = (price.iloc[-1] < sup_lvl) & (last_max_idx > df_mins['global_min'].cumsum().idxmax())
        df_maxs.loc[last_max_idx, 'global_max'] = True if cond else False

    global_dates = {
        'sell': df_maxs[df_maxs['global_max']].index,
        'buy': df_mins[df_mins['global_min']].index
    }
    res = []
    for action, action_date_idx in global_dates.items():
        shifted_action_date_idx = np.where(price.index.isin(action_date_idx))[0] + shift_days
        action_ser = pd.Series(
            [action for i in range(len(shifted_action_date_idx))]
        )
        to_concat = [
            price.iloc[shifted_action_date_idx].reset_index(), action_ser
        ]
        res.append(
            pd.concat(to_concat, axis=1)\
                .set_index('date')\
                .set_axis(['value', 'action'], axis=1)
        )

    fig, ax = plt.subplots(2, figsize=(12, 8))
    ax[0].plot(price)
    if mean_price_incr > 0:
        ax[0].scatter(price.index[-1], sup_lvl, c='orange')
    ax[0].scatter(df_maxs['max'].index, df_maxs['max'].values, s=40, c='g', label='sell')
    ax[0].scatter(df_mins['min'].index, df_mins['min'].values, s=40, c='r', label='buy')
    ax[0].set_title(f'Точки входа/выхода с частотой {freq}')
    ax[0].legend(fontsize=13)

    min_max = position_rotator(price, freq, shift_days, mode=mode)
    sell = min_max.loc[min_max['action'] == 'sell', 'value']
    buy = min_max.loc[min_max['action'] == 'buy', 'value']

    ax[1].plot(price)
    ax[1].scatter(sell.index, sell.values, s=40, c='g', label='sell')
    ax[1].scatter(buy.index, buy.values, s=40, c='r', label='buy')
    ax[1].set_title(f'Точки входа/выхода с частотой {freq}')
    ax[1].legend(fontsize=13)
    plt.show()


def plot_endog_vs_exog_index(data, return_cols=None, in_pycharm=True):
    if return_cols is None:
        all_return_cols = [col + '_return' for col in EXOG_DATA_PRICE_FILES + EXOG_DATA_OTHER_FILES]
        return_cols = list(filter(lambda x: x in data.columns, all_return_cols))
    if in_pycharm:
        for i, col in enumerate(return_cols):
            plt.figure(figsize=(12, 8))
            (100 * (1 + data[col]).cumprod()).plot(label=col)
            (100 * (1 + data['return']).cumprod()).plot(label='return')
            plt.legend()
    else:
        fig, ax = plt.subplots(len(return_cols), figsize=(8 * len(return_cols), 12))
        for i, col in enumerate(return_cols):
            (100 * (1 + data[col]).cumprod()).plot(label=col, ax=ax[i])
            (100 * (1 + data['return']).cumprod()).plot(label='return', ax=ax[i])
            ax[i].legend()


if __name__ == '__main__':
    from utils import PROJECT_ROOT

    data = pd.read_excel(PROJECT_ROOT / 'data/mcftrr.xlsx', index_col=[0], parse_dates=True)
    price = data['price'].copy()
    # plot_train_position_rotator(price, [42])
    plot_position_rotator(price, freqs=[42])
    a=1