from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TradingStrategyAnalyzer:
    """
    Класс для анализа торговых стратегий с использованием:
    - CUSUM-фильтра для обнаружения событий
    - Динамических ценовых барьеров
    - Генерации торговых меток
    - Визуализации результатов

    Параметры
    ----------
    price_series : pd.Series
        Временной ряд цен актива с датами в индексе
    """

    def __init__(self, price_series: pd.Series):
        self.price_series = price_series.copy()
        self.barriers_df = None
        self.labels = None

    def calculate_cusum_events(self, h: float) -> pd.DatetimeIndex:
        """
        Реализует CUSUM-фильтр для обнаружения значительных изменений в ценовом ряду.

        Parameters
        ----------
        h : float
            Порог для триггера событий

        Returns
        -------
        pd.DatetimeIndex
            Даты значительных ценовых движений
        """
        dt_events, s_pos, s_neg = [], 0, 0
        price_diff = self.price_series.diff()

        for dt, diff in zip(price_diff.index[1:], price_diff[1:].to_numpy()):
            s_pos = max(0, s_pos + diff)
            s_neg = min(0, s_neg + diff)

            if s_neg < -h:
                s_neg = 0
                dt_events.append(dt)
            elif s_pos > h:
                s_pos = 0
                dt_events.append(dt)

        return pd.DatetimeIndex(dt_events)

    def calculate_price_barriers(
        self, shift_days: int = 10, vol_span: int = 10, volatility_multiplier: int = 1
    ) -> pd.DataFrame:
        """
        Рассчитывает динамические ценовые барьеры.

        Parameters
        ----------
        shift_days : int, optional (default=10)
            Период для расчета исторической волатильности
        vol_span : int, optional (default=10)
            Период сглаживания волатильности
        volatility_multiplier : int, optional (default=1)
            Множитель для определения ширины канала

        Returns
        -------
        pd.DataFrame
            DataFrame с барьерами и метаданными
        """
        df = pd.DataFrame(self.price_series).reset_index()
        df["shift"] = df["price"].shift(shift_days)

        log_returns = np.log(df["price"] / df["shift"])
        df["volatility"] = log_returns.ewm(span=vol_span).std()

        df["upper_barrier"] = df["price"] * np.exp(
            volatility_multiplier * df["volatility"]
        )
        df["lower_barrier"] = df["price"] * np.exp(
            -volatility_multiplier * df["volatility"]
        )
        df["vertical_barrier"] = df["date"].shift(-shift_days)

        self.barriers_df = df.set_index("date")
        return self.barriers_df

    def generate_trading_labels(
        self, events: Optional[pd.DatetimeIndex] = None
    ) -> pd.Series:
        """
        Генерирует торговые метки на основе ценовых барьеров.

        Parameters
        ----------
        events : pd.DatetimeIndex, optional
            Даты для анализа. Если не указаны - используются все данные

        Returns
        -------
        pd.Series
            Торговые метки (1 - покупка, -1 - продажа)
        """
        if self.barriers_df is None:
            raise ValueError("Сначала вызовите calculate_price_barriers()")

        labels = []
        valid_indices = []

        processing_data = (
            self.barriers_df.loc[events] if events is not None else self.barriers_df
        )

        for idx, row in processing_data.iterrows():
            if pd.isna(row["vertical_barrier"]):
                continue

            try:
                price_window = self.barriers_df.loc[idx : row["vertical_barrier"]][
                    "price"
                ]
            except KeyError:
                continue

            upper_touch = price_window[price_window > row["upper_barrier"]].index.min()
            lower_touch = price_window[price_window < row["lower_barrier"]].index.min()
            vertical_limit = row["vertical_barrier"]

            first_touch = min(
                [
                    t
                    for t in [upper_touch, lower_touch, vertical_limit]
                    if t is not None and t in self.barriers_df.index
                ]
            )

            try:
                label = (
                    1
                    if (
                        self.barriers_df.at[first_touch, "price"]
                        > self.barriers_df.at[idx, "price"]
                    )
                    else -1
                )
                labels.append(label)
                valid_indices.append(idx)
            except KeyError:
                continue

        self.labels = pd.Series(
            labels, index=pd.DatetimeIndex(valid_indices), name="labels"
        )
        return self.labels

    def visualize_strategy(self, last_n_days: int = 365) -> None:
        """
        Визуализирует ценовой ряд с торговыми сигналами с обработкой временных меток.
        """
        if self.barriers_df is None or self.labels is None:
            raise ValueError(
                "Сначала вызовите calculate_price_barriers() и generate_trading_labels()"
            )

        # Получаем последние N дней через .loc
        end_date = self.barriers_df.index.max()
        start_date = end_date - pd.DateOffset(days=last_n_days)
        plot_data = self.barriers_df.loc[start_date:end_date]

        # Фильтруем метки, оставляя только существующие в plot_data
        valid_labels = self.labels[self.labels.index.isin(plot_data.index)]

        # Проверяем наличие данных для отображения
        if plot_data.empty or valid_labels.empty:
            print("Нет данных для визуализации в указанном диапазоне")
            return

        plt.figure(figsize=(14, 7))
        plt.plot(plot_data.index, plot_data["price"], label="Price", alpha=0.7)

        # Разделяем сигналы с проверкой наличия в данных
        buy_signals = valid_labels[valid_labels == 1]
        sell_signals = valid_labels[valid_labels == -1]

        if not buy_signals.empty:
            plt.scatter(
                buy_signals.index,
                plot_data.loc[buy_signals.index, "price"],
                color="green",
                marker="^",
                s=80,
                label="Buy",
                edgecolors="black",
            )

        if not sell_signals.empty:
            plt.scatter(
                sell_signals.index,
                plot_data.loc[sell_signals.index, "price"],
                color="red",
                marker="v",
                s=80,
                label="Sell",
                edgecolors="black",
            )

        # Форматирование графика
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.grid(alpha=0.3)
        plt.title(f"Trading Signals ({len(valid_labels)} signals)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_analysis_report(self) -> dict:
        """
        Формирует отчет о результатах анализа.

        Returns
        -------
        dict
            Статистика по сгенерированным сигналам
        """
        if self.labels is None:
            return {}

        return {
            "total_signals": len(self.labels),
            "buy_signals": sum(self.labels == 1),
            "sell_signals": sum(self.labels == -1),
            "signal_frequency": len(self.labels) / len(self.price_series),
        }


def main():
    price_data = pd.read_excel("data/endog_data/mcftrr.xlsx", index_col="date")
    # Инициализация анализатора
    analyzer = TradingStrategyAnalyzer(price_data)

    # Расчет событий и барьеров
    events = analyzer.calculate_cusum_events(h=100)
    barriers = analyzer.calculate_price_barriers(
        shift_days=10, vol_span=20, volatility_multiplier=2
    )

    # Генерация меток
    labels = analyzer.generate_trading_labels(events=events)
    # labels = analyzer.generate_trading_labels(events=None)

    # Визуализация и отчет
    analyzer.visualize_strategy()
    print(analyzer.get_analysis_report())


if __name__ == "__main__":
    main()
