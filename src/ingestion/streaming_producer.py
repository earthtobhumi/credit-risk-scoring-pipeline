"""
Simulates a live stream of incoming loan applications.

This is a drop-in stand-in for a Kafka producer: it writes one JSON record
per line to a local "topic" file, with a small delay between records, the
same way a real producer would push events onto a Kafka topic in real time.

SWAPPING IN REAL CONFLUENT KAFKA:
    1. `docker-compose -f docker-compose.kafka.yml up -d` (or use Confluent Cloud
       free tier and put the bootstrap servers / API key in .env)
    2. Replace the `open(STREAM_TOPIC_FILE, "a")` write below with:
           from confluent_kafka import Producer
           producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS, ...})
           producer.produce("loan-applications", json.dumps(record).encode())
           producer.flush()
    3. streaming_consumer.py's `tail` loop becomes a Kafka Consumer.poll() loop.
    Nothing else in the pipeline (Bronze write, schema) needs to change --
    consumers only care that JSON records arrive, not how.

Usage:
    python -m src.ingestion.streaming_producer --n-events 20 --delay 0.5
"""
import argparse
import json
import time
import datetime as dt

import pandas as pd

from src.common.config import RAW_APPLICATIONS_CSV, STREAM_TOPIC_FILE


def produce(n_events: int = 20, delay: float = 0.5):
    df = pd.read_csv(RAW_APPLICATIONS_CSV)
    sample = df.sample(n=min(n_events, len(df)), random_state=None)

    with open(STREAM_TOPIC_FILE, "a") as topic:
        for _, row in sample.iterrows():
            record = row.to_dict()
            record["_event_ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
            topic.write(json.dumps(record, default=str) + "\n")
            topic.flush()
            print(f"produced application SK_ID_CURR={record['SK_ID_CURR']}")
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    produce(args.n_events, args.delay)
