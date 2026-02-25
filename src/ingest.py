import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH, RAW_DIR


@dataclass(frozen=True)
class IngestConfig:
    n_rows: int = 5000
    random_state: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_churn_dataset(n_rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], size=n_rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.6, 0.25, 0.15],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.45, 0.15, 0.2, 0.2],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "None"],
        size=n_rows,
        p=[0.35, 0.5, 0.15],
    )

    online_security = rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65])
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])

    age = rng.integers(18, 80, size=n_rows)

    tenure = rng.integers(0, 72, size=n_rows)

    base_monthly = rng.normal(loc=70, scale=20, size=n_rows).clip(20, 130)

    fiber_surcharge = (internet_service == "Fiber optic").astype(int) * rng.normal(
        loc=18, scale=6, size=n_rows
    )
    dsl_surcharge = (internet_service == "DSL").astype(int) * rng.normal(
        loc=6, scale=3, size=n_rows
    )
    none_discount = (internet_service == "None").astype(int) * rng.normal(
        loc=-20, scale=4, size=n_rows
    )

    security_discount = (online_security == "Yes").astype(int) * rng.normal(
        loc=-5, scale=2, size=n_rows
    )
    support_discount = (tech_support == "Yes").astype(int) * rng.normal(
        loc=-4, scale=2, size=n_rows
    )

    monthly_charges = (base_monthly + fiber_surcharge + dsl_surcharge + none_discount + security_discount + support_discount).clip(
        18, 160
    )

    total_charges = (monthly_charges * tenure + rng.normal(loc=0, scale=50, size=n_rows)).clip(
        0, None
    )

    # --- churn logic: stronger signals so models can learn well ---
    contract_score = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [1.6, 0.0, -1.2],
        default=0.0,
    )

    payment_score = np.select(
        [payment_method == "Electronic check", payment_method == "Mailed check"],
        [0.8, 0.2],
        default=-0.3,
    )

    service_score = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "None"],
        [0.6, 0.0, -0.6],
        default=0.0,
    )

    support_score = np.where(tech_support == "No", 0.5, -0.5)
    security_score = np.where(online_security == "No", 0.4, -0.4)

    tenure_score = -0.06 * tenure
    charges_score = 0.015 * (monthly_charges - 70)
    age_score = -0.008 * (age - 40)

    logit = (
        -0.5
        + contract_score
        + payment_score
        + service_score
        + support_score
        + security_score
        + tenure_score
        + charges_score
        + age_score
        + rng.normal(0, 0.3, size=n_rows)  # small noise for realism
    )

    churn_prob = _sigmoid(logit)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "gender": gender,
            "contract": contract,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "churn": churn,
        }
    )

    return df


def save_raw_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def ingest_to_duckdb(df: pd.DataFrame, db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE churn AS SELECT * FROM df")
    con.execute("CREATE OR REPLACE VIEW churn_summary AS SELECT churn, COUNT(*) AS n FROM churn GROUP BY churn")
    con.close()


def main() -> None:
    cfg = IngestConfig()

    raw_csv_path = os.path.join(RAW_DIR, "churn.csv")

    df = generate_churn_dataset(n_rows=cfg.n_rows, random_state=cfg.random_state)
    save_raw_csv(df, raw_csv_path)
    ingest_to_duckdb(df, DB_PATH)

    churn_rate = df["churn"].mean()
    print(f"Generated {len(df):,} rows")
    print(f"Raw CSV: {raw_csv_path}")
    print(f"DuckDB: {DB_PATH}")
    print(f"Churn rate: {churn_rate:.3f}")


if __name__ == "__main__":
    main()
