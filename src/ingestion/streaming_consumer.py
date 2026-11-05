"""
Consumes events from the simulated stream topic and appends them to the
Bronze Delta table -- the real-time counterpart to batch_loader.py.

Tracks a byte offset so it only reads new lines since the last run
(a crude stand-in for a Kafka consumer group's committed offset).

Usage:
    python -m src.ingestion.streaming_consumer
"""
import json
import datetime as dt

import pandas as pd
from deltalake import write_deltalake

from src.common.config import STREAM_TOPIC_FILE, BRONZE_APPLICATIONS_TABLE, DATA_DIR

OFFSET_FILE = DATA_DIR / ".stream_consumer_offset"


def _read_offset() -> int:
    if OFFSET_FILE.exists():
        return int(OFFSET_FILE.read_text().strip() or 0)
    return 0


def _write_offset(offset: int):
    OFFSET_FILE.write_text(str(offset))


def consume() -> int:
    if not STREAM_TOPIC_FILE.exists():
        print("No stream topic file yet -- run streaming_producer.py first.")
        return 0

    offset = _read_offset()
    records = []
    with open(STREAM_TOPIC_FILE, "rb") as f:
        f.seek(offset)
        for line in f:
            if line.strip():
                records.append(json.loads(line))
        new_offset = f.tell()

    if not records:
        print("No new events since last consume.")
        return 0

    df = pd.DataFrame(records)
    df["_ingested_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    df["_source"] = "stream_simulated"

    write_deltalake(BRONZE_APPLICATIONS_TABLE, df, mode="append", schema_mode="merge")
    _write_offset(new_offset)
    print(f"Consumed {len(df)} streamed events into Bronze")
    return len(df)


if __name__ == "__main__":
    consume()
