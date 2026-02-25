import os
from dataclasses import dataclass

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DB_PATH, PROCESSED_DIR


@dataclass(frozen=True)
class PreprocessConfig:
    test_size: float = 0.2
    random_state: int = 42
    target_col: str = "churn"


CATEGORICAL_COLS = [
    "gender",
    "contract",
    "payment_method",
    "internet_service",
    "online_security",
    "tech_support",
]

NUMERIC_COLS = ["age", "tenure", "monthly_charges", "total_charges"]


def load_from_duckdb(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM churn").fetchdf()
    con.close()
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure"])

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["churn"] = df["churn"].astype(int)

    return df


def split_and_save(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[str, str]:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_col],
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def main() -> None:
    cfg = PreprocessConfig()

    df = load_from_duckdb(DB_PATH)
    df = basic_cleaning(df)

    train_path, test_path = split_and_save(df, cfg)

    print(f"Loaded {len(df):,} rows from DuckDB")
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")


if __name__ == "__main__":
    main()
