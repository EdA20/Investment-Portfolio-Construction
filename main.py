import argparse
import logging
import os
import sys
from warnings import simplefilter

import numpy as np
import pandas as pd

from portfolio_constructor.feature_filter import feature_filter
from portfolio_constructor.feature_generator import data_generator, replace_old_feature_names
from portfolio_constructor.sampler import SampleStrategy
from portfolio_constructor.model import read_logger, strategy_full_cycle


def setup_main_logger():
    # Создаём логгер (не используем basicConfig!)
    logger = logging.getLogger("main")  # Уникальное имя
    logger.setLevel(logging.INFO)

    # Удаляем старые обработчики (если были)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Настройка консольного вывода
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)
    logger.propagate = False  # Чтобы не дублировалось в корневой логгер

    return logger


def main(parser):
    main_logger.info("Старт")
    is_debug_mode = getattr(sys, "gettrace", lambda: None)() is not None

    if is_debug_mode:
        args = argparse.Namespace(features="best", filter=False, sampling=False)
    else:
        args = parser.parse_args()

    if "data.parquet" in os.listdir("data"):
        main_logger.info("Загрузка данных из data.parquet")
        data = pd.read_parquet("data/data.parquet")
    else:
        main_logger.warning("Файл data.parquet не найден. Генерация новых данных")
        data = data_generator(path="mcftrr.xlsx")
        data.to_parquet("data/data.parquet")

    if args.features == "all":
        features = list(data.drop(["price", "ruonia_daily"], axis=1).columns)

    elif args.features == "random":
        features = list(data.drop(["price", "ruonia_daily"], axis=1).columns)
        features = list(np.random.choice(features, 20, replace=False))

    elif args.features == "best":
        features = ('long/short_physic_ratio_return_sw_std_10_21', 'bonds10y_sw_skew_10/21', 'top3_mean_vol_sw_skew_10/63', 'imoex_pe_return_sw_kurt_10', 'short_physic_ew_std_5/21', 'short_physic_sw_skew_10', 'top3_mean_vol_sw_std_21/126', 'long/short_entity_ratio_return_roc_252', 'long_physic_return_sw_std_63/126', 'short_physic_sw_kurt_10/21', 'long/short_entity_ratio_return_sw_mean_10/21', 'min/price_63', 'price_return_ew_mean_5/21', 'oscillator_sw_63', 'imoex_pe_sw_kurt_5')

    elif args.features == 'noise':
        data['noise'] = np.random.normal(size=(len(data), 20))

    if args.filter:
        features = feature_filter(data, var_threshold=0.001, corr_threshold=0.85)

    step = 20
    splitter_kwargs = dict(
        forecast_horizon=[i for i in range(1, step + 1)],
        step_length=step,
        initial_window=500,
        window_length=None,
        eval_obs=0,
        set_sample_weights=True,
    )
    position_rotator_kwargs = dict(
        markup_name='triple_barrier',
        markup_kwargs=dict(
            h=250, shift_days=63, vol_span=20, volatility_multiplier=2
        ),
        # markup_name='min_max',
        # markup_kwargs=dict(
        #     freq=63, rotator_type=1, inplace=True
        # )
    )
    sample_weight_kwargs = dict(
        weight_params=dict(
            time_critical=dict(
                k=100,
                q=0.01,
                jump_coef=20,
                fading_factor=21
            )
        )
    )
    model_kwargs = dict(
        model_name="catboost",
        n_models=5,
        iterations=10,
        subsample=0.8,
        random_state=2,
    )
    strat_kwargs = dict(
        prob_to_weight=True, weight_prob_threshold=0.5
    )
    plot_kwargs = dict(
        perf_plot=True, sliding_plot=True, save=False  # сразу рисует динамику портфеля
    )

    if not args.sampling:
        main_logger.info("Старт обучения модели")
        output = strategy_full_cycle(
            data,
            features,
            splitter_kwargs,
            sample_weight_kwargs,
            position_rotator_kwargs,
            model_kwargs,
            strat_kwargs,
            plot_kwargs,
            # val_date_breakpoint='2024-01-01'
        )
        main_logger.info("Обучение завершено")
    else:
        sampler = SampleStrategy(
            data,
            features,
            splitter_kwargs,
            sample_weight_kwargs,
            position_rotator_kwargs,
            model_kwargs,
            plot_kwargs,
            prob_to_weight=True
        )
        random_seed_res = sampler.random_features_sampling(
            features, max_amount_features=30, n_trials=2, seed_sampling=2
        )
        a=1


if __name__ == "__main__":
    # Добавить в начало файла
    main_logger = setup_main_logger()
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument(
        "--features",
        choices=["all", "important", "last_best", "random"],
        default="all",
        help='whether to use all features ("all") or features from features_importance.xlsx ("important")',
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="variance and correlation feature filtering",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="whether to sample strategies with different feature subsets or sample one strategy with all features",
    )
    simplefilter("ignore")
    main(parser)


# зафитить деревья для каждого признака отдельно -> выбрать топ N по метрике