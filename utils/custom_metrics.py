import numpy as np

__all__ = ['ProfitMetric', 'CriticalF1Metric', 'FocalLossMetric', 'FocalLossObjective']

MAX_Z = 33


class ProfitMetric(object):
    def __init__(self, profit_metric_df):
        self.profit_metric_df = profit_metric_df

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]
        val_data_idx = np.where(self.profit_metric_df['size'] == len(approx))[0][0]
        val_data = self.profit_metric_df.loc[val_data_idx, 'data'].copy()

        val_data['preds'] = np.exp(approx) / (1 + np.exp(approx))
        val_data['preds'] = val_data['preds'].shift().fillna(0)

        val_data['moex_long_weight'] = val_data['preds'].apply(lambda x: x if x >= 0.5 else 0)
        val_data['strat_return'] = (
            val_data['moex_long_weight'] * val_data['price_return'] +
            (1 - val_data['moex_long_weight']) * val_data['ruonia_daily']
        )
        val_data.loc[0, ['price_return', 'strat_return']] = 0
        val_data['strategy_perf'] = (val_data['strat_return'] + 1).cumprod() * 100
        val_data['bench_perf'] = (val_data['strat_return'] + 1).cumprod() * 100
        val_data['perf_delta'] = val_data['strategy_perf'] - val_data['bench_perf']
        median_perf_delta = val_data['perf_delta'].median()

        return median_perf_delta, 1


class CriticalF1Metric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes - предсказания модели
        # target - истинные значения
        # weight - веса наблюдений

        # Выделяем критичные наблюдения
        critical_mask = (weight == 10)  # Предполагаем, что вес=10 для важных точек
        approx = np.array(approxes[0])
        y_pred = (approx > 0.5).astype(int)
        y_true = np.array(target)

        # Расчет F1 только на критичных данных
        y_true_critical = y_true[critical_mask]
        y_pred_critical = y_pred[critical_mask]

        tp = np.sum((y_true_critical == 1) & (y_pred_critical == 1))
        fp = np.sum((y_true_critical == 0) & (y_pred_critical == 1))
        fn = np.sum((y_true_critical == 1) & (y_pred_critical == 0))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        return f1, 1  # Возвращаем F1 и вес (не используется)


class FocalLossObjective:
    def calc_ders_range(self, approxes, targets, weights):
        ders1, ders2 = [], []

        alpha = 0.5
        gamma = 1.5

        for i in range(len(approxes)):
            z = approxes[i]
            y = targets[i]
            w = weights[i] if weights is not None else 1.0

            z = MAX_Z if z > MAX_Z else (MAX_Z if z >= -MAX_Z else -MAX_Z)
            p = 1 / (1 + np.exp(-z))

            if y == 1:
                # Градиент и гессиан для y=1 (проверено ранее)
                grad = alpha * (1 - p)**gamma * (gamma * p * np.log(p) + p - 1)
                hess = alpha * p * (1 - p)**(1 + gamma) * (
                    1 + 2 * gamma -
                    gamma * np.log(p) -
                    (gamma**2 * p * np.log(p)) / (1 - p)
                )
            else:
                # Градиент и гессиан для y=0 (на основе Wolfram Alpha)
                grad = (1 - alpha) * p**gamma * (gamma * (1 - p) * np.log(1 - p) - p)
                hess = -(1 - alpha) * p**gamma * (1 - p) * (
                    gamma * np.log(1 - p) * (gamma * p - gamma + p) +
                    p * (2 * gamma + 1)
                )

            ders1.append(w * grad)
            ders2.append(w * hess)

        return list(zip(ders1, ders2))


class FocalLossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        alpha = 0.5
        gamma = 1.5

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            z = approx[i]
            t = target[i]

            z = MAX_Z if z > MAX_Z else (z if z >= -MAX_Z else -MAX_Z)
            p = 1 / (1 + np.exp(-z))

            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += -w * (
                alpha * np.log(1-p)**gamma * t * np.log(p) +
                (1 - alpha) * np.log(p)**gamma * (1 - t) * np.log(1 - p)
            )

        return error_sum, weight_sum
