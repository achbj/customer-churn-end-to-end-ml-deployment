"""Run the full pipeline: ingest -> preprocess -> train."""

from src.ingest import main as ingest
from src.preprocess import main as preprocess
from src.train import train_and_evaluate


def main():
    print("=" * 50)
    print("Step 1: Data Ingestion")
    print("=" * 50)
    ingest()

    print("\n" + "=" * 50)
    print("Step 2: Preprocessing")
    print("=" * 50)
    preprocess()

    print("\n" + "=" * 50)
    print("Step 3: Model Training")
    print("=" * 50)
    train_and_evaluate()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
