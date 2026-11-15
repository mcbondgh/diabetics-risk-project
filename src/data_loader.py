import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = config.FEATURES + [config.TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[needed].dropna()


def summarize_data(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nClass balance:")
    print(df[config.TARGET].value_counts(normalize=True).round(3))


def split_data(df: pd.DataFrame):
    X = df.loc[:, config.FEATURES]
    y = df[config.TARGET].astype(int)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )
