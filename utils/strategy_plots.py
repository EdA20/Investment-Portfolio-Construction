import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

__all__ = [
    'plot_sliding_start_pnl',
    'plot_startegy_performance',
    'plot_shifted_strategy_with_benchmark',
    'plot_random_features_pnl'
]


def plot_sliding_start_pnl(res, days=365*3, plot=None):

    cols = res.columns
    if 'strat_return' not in cols:
        raise Exception('input dataframe must contain "strat_return" column')
    if 'price_return' not in cols:
        raise Exception('input dataframe must contain "price_return" column')

    res[f'strat_hist_pnl_{days}d'] = res['strat_return'].rolling(f'{days}D').apply(lambda x: 100 * (1 + x).prod() - 100)
    res[f'moex_hist_pnl_{days}d'] = res['price_return'].rolling(f'{days}D').apply(lambda x: 100 * (1 + x).prod() - 100)

    if plot:
        plt.figure(figsize=(10,6), dpi=100)
        if plot == 'hist':
            plt.hist(res[f'strat_hist_pnl_{days}d'], alpha=0.4, label='ML strat')
            plt.hist(res[f'moex_hist_pnl_{days}d'], alpha=0.4, label='MOEX')
        else:
            plt.plot(res[f'strat_hist_pnl_{days}d'], label='ML strat')
            plt.plot(res[f'moex_hist_pnl_{days}d'], label='MOEX')
        plt.legend()
        plt.title(f'ML strat vs IMOEX {days} days PnL')
        plt.show()

    return res


def plot_startegy_performance(res):

    cols = res.columns
    if 'strategy_perf' not in cols:
        raise Exception('input dataframe must contain "strategy_perf" column')
    if 'bench_perf' not in cols:
        raise Exception('input dataframe must contain "bench_perf" column')

    plt.figure(figsize=(15,8), dpi=100)
    plt.plot(res['strategy_perf'], label='Strategy Performance')
    plt.plot(res['bench_perf'], label='Benchmark Performance')

    plt.legend()
    plt.title('Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Performance')
    plt.grid(True)

    plt.show()


def plot_shifted_strategy_with_benchmark(data, sample_strategy_returns):

    sample_strategy_returns = sample_strategy_returns.copy()
    sample_strategy_returns.iloc[0] = 0
    sample_strategy_returns = (sample_strategy_returns + 1).cumprod() * 100

    moex_perf = data.loc[sample_strategy_returns.index, 'price_return']
    moex_perf.iloc[0] = 0
    moex_perf = (1 + moex_perf).cumprod() * 100

    pnls = sample_strategy_returns.iloc[-1].sort_values()

    quantiles = [0.05, 0.5, 0.95]
    compare_pnl = {'bench': moex_perf}
    for q in quantiles:
        q_iloc = [int(len(pnls) * q)]
        col_name = pnls.iloc[q_iloc].index[0]
        compare_pnl[q] = sample_strategy_returns[col_name]
    compare_pnl = pd.DataFrame(compare_pnl)

    plt.figure(figsize=(15,8), dpi=100)
    for q in quantiles:
        plt.plot(
            compare_pnl[q],
            label=f'{q} квантиль распределения PnL',
            alpha=1 if q == 0.5 else 0.5,
            ls='-' if q == 0.5 else '--'
        )
    plt.plot(compare_pnl['bench'], label=f'bench')
    plt.legend(fontsize=12)
    plt.title(f'PnL стратегий', fontsize=15)
    plt.show()


def plot_random_features_pnl(random_features_pnl):
    cols = random_features_pnl.columns
    if len(cols) == 4:
        fig, axes = plt.subplots(2, figsize=(8, 10), dpi=100)
    else:
        fig, axes = plt.subplots(1, dpi=100)
        axes = [axes]

    for i, col in enumerate(['mean_median_perf_delta', 'mean_std_perf_delta'][:len(axes)]):
        ax = axes[i]
        sc = ax.scatter(
            random_features_pnl['pnl_std'], random_features_pnl['pnl_mean'],
            c=random_features_pnl[col], cmap='inferno'
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax, label=f'Mean {col.split("_")[1].title()} Performance Delta')\
            .set_label(f'Mean {col.split("_")[1].title()} Performance Delta', rotation=270, labelpad=25)

        ax.set_xlabel('std PnL')
        ax.set_ylabel('mean PnL')
        ax.set_title('PnL sample with 5 seeds and random features')

    plt.tight_layout()
    plt.show()
