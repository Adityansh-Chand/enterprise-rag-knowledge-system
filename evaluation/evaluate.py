
from pathlib import Path

import pandas as pd

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "sample_data.csv"
df = pd.read_csv(DATASET)

print("records:", len(df))
