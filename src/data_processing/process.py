from typing import Optional
from data_ingestion.kafka_pipeline import KafkaSSHClient
import os
from dotenv import load_dotenv
from kafka.errors import NoBrokersAvailable
import re
import csv
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


# define a rotating csv writer class that appends to a csv file and rotates it when it reaches a certain size
class RotatingCSVWriter:
    """
    Timestamp-based CSV rotator:
      - Filenames: <stem>_<YYYYMMDD-HHMMSS>[optional _N]<ext>
      - On startup: finds the most recent existing file (by timestamp),
        counts rows, and appends until threshold is hit, then rotates.
    """

    TS_FMT = "%Y%m%d-%H%M%S"

    def __init__(self, base_filename: str, fieldnames: List[str], threshold: int):
        self.base_filename = base_filename  # e.g., ".../movie_log.csv"
        self.fieldnames = fieldnames
        self.threshold = threshold

        # Ensure directory exists
        out_dir = os.path.dirname(self.base_filename) or "."
        os.makedirs(out_dir, exist_ok=True)

        self.stem, self.ext = os.path.splitext(self.base_filename)
        self.current_path, self.row_count = self._init_from_disk()
        self._ensure_header()

    # ---------- helpers ----------
    def _ts_tag(self) -> str:
        return datetime.now().strftime(self.TS_FMT)

    def _path_for_tag(self, tag: str, suffix_index: int = 1) -> str:
        # suffix_index==1 → no _N; >=2 → _N
        suffix = "" if suffix_index == 1 else f"_{suffix_index}"
        return f"{self.stem}_{tag}{suffix}{self.ext}"

    def _glob_pattern(self) -> str:
        # matches base stem with any _timestamp[_N]
        return f"{self.stem}_*{self.ext}"

    def _parse_name(self, path: str) -> Tuple[datetime, int] | None:
        """
        Extract (timestamp_dt, dup_index) from filename.
        Accepts either: stem_YYYYMMDD-HHMMSS.ext  or stem_YYYYMMDD-HHMMSS_N.ext
        """
        name = os.path.basename(path)
        # Escape stem and ext for regex
        stem_esc = re.escape(os.path.basename(self.stem))
        ext_esc = re.escape(self.ext)
        m = re.match(rf"^{stem_esc}_(\d{{8}}-\d{{6}})(?:_(\d+))?{ext_esc}$", name)
        if not m:
            return None
        ts_str, dup = m.groups()
        try:
            dt = datetime.strptime(ts_str, self.TS_FMT)
        except ValueError:
            return None
        dup_idx = int(dup) if dup else 1
        return (dt, dup_idx)

    def _list_existing_sorted(self) -> list[str]:
        paths = glob.glob(self._glob_pattern())
        parsed = []
        for p in paths:
            meta = self._parse_name(p)
            if meta:
                parsed.append((meta[0], meta[1], p))
        # sort by timestamp then duplicate index
        parsed.sort(key=lambda x: (x[0], x[1]))
        return [p for *_, p in parsed]

    def _count_rows_no_header(self, path: str) -> int:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return 0
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            total = sum(1 for _ in reader)
        return max(0, total - 1)  # minus header

    def _ensure_header(self):
        new_file = (
            not os.path.exists(self.current_path)
            or os.path.getsize(self.current_path) == 0
        )
        if new_file:
            with open(self.current_path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def _new_unused_path(self) -> str:
        """
        Build a new path with current timestamp; if it already exists (same second),
        increment _2, _3, ... until unused.
        """
        tag = self._ts_tag()
        idx = 1
        while True:
            candidate = self._path_for_tag(tag, idx)
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def _init_from_disk(self) -> tuple[str, int]:
        existing = self._list_existing_sorted()
        if not existing:
            # no prior files → start fresh
            return self._new_unused_path(), 0

        last_path = existing[-1]
        rows = self._count_rows_no_header(last_path)
        if rows >= self.threshold:
            # last file is full → start a new one
            return self._new_unused_path(), 0
        else:
            # continue writing to last file
            return last_path, rows

    def _rotate(self):
        self.current_path = self._new_unused_path()
        self.row_count = 0
        self._ensure_header()

    # ---------- public ----------
    def write_row(self, row: dict):
        if self.row_count >= self.threshold:
            self._rotate()
        with open(self.current_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
        self.row_count += 1


class DataProcessor:
    def __init__(self, kafka_client: KafkaSSHClient):
        self.kafka_client = kafka_client

        # 1) Set your desired output directory (absolute path you gave)
        OUTPUT_DIR = Path(
            "/Users/macbookprom1/Downloads/CMU/CMU_FALL_25/ML_PROD/PROJECT/group-project-f25-home-alone/src/data_processing/output"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists
        MOVIE_DIR = OUTPUT_DIR / "movie"
        MOVIE_DIR.mkdir(parents=True, exist_ok=True)
        RATING_DIR = OUTPUT_DIR / "rating"
        RATING_DIR.mkdir(parents=True, exist_ok=True)
        RECO_DIR = OUTPUT_DIR / "recommendation"
        RECO_DIR.mkdir(parents=True, exist_ok=True)

        # Define schema (column order) per message type
        self.schemas = {
            "movie": ["type", "time", "userid", "movieid", "minute"],
            "rating": ["type", "time", "userid", "movieid", "rating"],
            "recommendation_request": [
                "type",
                "time",
                "userid",
                "server",
                "status",
                "recommendations",
                "responsetime",
            ],
        }

        # 2) Use the directory when setting base filenames
        self.writers = {
            "movie": RotatingCSVWriter(
                str(MOVIE_DIR / "movie_log.csv"),
                self.schemas["movie"],
                threshold=10_000,
            ),
            "rating": RotatingCSVWriter(
                str(RATING_DIR / "rating_log.csv"), self.schemas["rating"], threshold=50
            ),
            "recommendation_request": RotatingCSVWriter(
                str(RECO_DIR / "recommendation_log.csv"),
                self.schemas["recommendation_request"],
                threshold=50,
            ),
        }

        # Precompile regexes
        self.movie_pattern = re.compile(
            r"^([^,]+),([^,]+),GET /data/m/([^/]+)/([^\.]+)\.mpg$"
        )
        self.rating_pattern = re.compile(r"^([^,]+),([^,]+),GET /rate/([^=]+)=([^ ]+)$")
        self.reco_pattern = re.compile(
            r"^([^,]+),([^,]+),recommendation request ([^,]+), status (\d+), result: ([^,]+), ([^,]+)$"
        )

    def process_data(self, data: str) -> Optional[dict]:
        data = data.strip()

        m = self.movie_pattern.match(data)
        if m:
            time, userid, movieid, minute = m.groups()
            return {
                "type": "movie",
                "time": time,
                "userid": userid,
                "movieid": movieid,
                "minute": minute,
            }

        m = self.rating_pattern.match(data)
        if m:
            time, userid, movieid, rating = m.groups()
            return {
                "type": "rating",
                "time": time,
                "userid": userid,
                "movieid": movieid,
                "rating": rating,
            }

        m = self.reco_pattern.match(data)
        if m:
            time, userid, server, status, recommendations, responsetime = m.groups()
            return {
                "type": "recommendation_request",
                "time": time,
                "userid": userid,
                "server": server,
                "status": status,
                "recommendations": recommendations,
                "responsetime": responsetime,
            }

        # Unknown line format
        return None

    def run(self):
        for message in self.kafka_client.consume():
            processed = self.process_data(message)
            if not processed:
                # Skip unknown/unparseable messages
                print("Skipped (unrecognized format):", message)
                continue

            msg_type = processed["type"]
            writer = self.writers.get(msg_type)
            if not writer:
                print("No writer for type:", msg_type)
                continue

            # Ensure row has all fields in the defined schema/order
            ordered_row = {k: processed.get(k, "") for k in self.schemas[msg_type]}
            writer.write_row(ordered_row)

            print(f"Wrote {msg_type} record.")


if __name__ == "__main__":
    load_dotenv()

    client = KafkaSSHClient(
        username=os.getenv("TUNNEL_USER"),
        host=os.getenv("TUNNEL_IP"),
        password=os.getenv("TUNNEL_PASSWORD"),
        local_port=os.getenv("TUNNEL_PORT"),
        remote_port=os.getenv("TUNNEL_PORT"),
    )

    # Consume from topic "movielog10"

    while True:
        try:
            data_processor = DataProcessor(kafka_client=client)
            data_processor.run()
        except NoBrokersAvailable:
            print(
                "No Kafka brokers available. Please check your connection and broker settings."
            )
        except KeyboardInterrupt:
            print("Stopping consumption...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
