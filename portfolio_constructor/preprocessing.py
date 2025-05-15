import pandas as pd

from portfolio_constructor import PROJECT_ROOT

RAW_FILES_FOLDER = PROJECT_ROOT / "data/exog_data/raw/"
PREPROCESSED_FILES_FOLDER = PROJECT_ROOT / "data/exog_data/preprocessed/"


def investing_ru_brent_file_preprocessing(file_name="brent.csv"):
    input_path = RAW_FILES_FOLDER / file_name
    output_path = PREPROCESSED_FILES_FOLDER / file_name

    df = pd.read_csv(input_path, parse_dates=[0], dayfirst=True)
    df["Цена"] = df["Цена"].str.replace(",", ".").astype(float)
    df = df.loc[:, ["Дата", "Цена"]]
    df = df.set_axis(["date", "value"], axis=1)
    df = df.sort_values("date", ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


def cbr_ruonia_file_preprocessing(file_name="ruonia.xlsx"):
    input_path = RAW_FILES_FOLDER / file_name
    output_path = PREPROCESSED_FILES_FOLDER / file_name

    df = pd.read_excel(input_path)
    df = df.loc[:, ["DT", "ruo"]].copy()
    df = df.set_axis(["date", "value"], axis=1)
    df = df.sort_values("date", ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


def cbr_usd_file_preprocessing(file_name="usd.xlsx"):
    input_path = RAW_FILES_FOLDER / file_name
    output_path = PREPROCESSED_FILES_FOLDER / file_name

    df = pd.read_excel(input_path)
    df = df.drop(["nominal", "cdx"], axis=1).rename(
        {"data": "date", "curs": "value"}, axis=1
    )
    df["date"] = df["date"] - pd.Timedelta(days=1)
    df = df.sort_values("date", ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


def cbr_gold_file_preprocessing(file_name="gold_rub.xlsx", usd_file_name="usd.xlsx"):
    input_path = RAW_FILES_FOLDER / file_name
    output_path = PREPROCESSED_FILES_FOLDER / file_name
    usd_path = PREPROCESSED_FILES_FOLDER / usd_file_name

    df = pd.read_excel(input_path)
    df["date"] = df["date"] - pd.Timedelta(days=1)
    usd = pd.read_excel(usd_path)
    df = pd.merge(df, usd, how="left", on="date", suffixes=("_gold", "_usd"))
    # перевожу из рубли за грамм в доллар за унцию
    df["value_gold"] = df["value_gold"] / df["value_usd"] * 28.35
    df = df.drop("value_usd", axis=1).set_axis(["date", "value"], axis=1)
    df = df.sort_values("date", ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


def moex_fut_pos_preprocessing(file_name="mix_pos_struct.xlsx", drop_cols=False):
    input_path = RAW_FILES_FOLDER / file_name
    output_path = PREPROCESSED_FILES_FOLDER / file_name

    df = pd.read_excel(input_path)
    df["long/short_entity_ratio"] = df["long_entity"] / df["short_entity"]
    df["long/short_physic_ratio"] = df["long_physic"] / df["short_physic"]
    if drop_cols:
        df = df[["date", "long/short_entity_ratio", "long/short_physic_ratio"]].copy()
    df = df.sort_values("date", ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


if __name__ == "__main__":
    moex_fut_pos_preprocessing()
