
import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sys
from pathlib import Path

import pandas as pd

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from services.data_pipeline import DataPipeline
from models.global_forecaster import GlobalForecaster


def main() -> int:
    pipeline = DataPipeline(data_dir=str(repo / "Data"))
    pipeline.load_all()
    pipeline.load_core_tables()

    oi = pipeline.order_items.copy()
    oi["date"] = pd.to_datetime(oi["created"], unit="s").dt.normalize()

    daily = oi.groupby(["date", "item_id"])["quantity"].sum().reset_index()
    daily.columns = ["date", "item_id", "order_count"]

    item_counts = daily.groupby("item_id")["order_count"].sum().sort_values(ascending=False)
    top_items = item_counts.head(50).index.tolist()
    daily_small = daily[daily["item_id"].isin(top_items)].copy()

    model = GlobalForecaster(
        recent_days=90,
        fill_missing_days=True,
        max_resampled_rows=300_000,
        verbose=False,
    )
    model.fit(daily_small)

    test_item = int(top_items[0])
    preds = model.predict(test_item, horizon_days=7)
    print("Pred rows:", len(preds))
    print(preds.head().to_string(index=False))

    preds2 = model.predict(999_999_999, horizon_days=3)
    print("Unseen rows:", len(preds2))
    print(preds2.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())