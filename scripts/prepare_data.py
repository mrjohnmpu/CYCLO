"""Verify complete published datasets and optionally unpack the training CSV."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(data_dir: Path, unpack: bool = False) -> list[dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    records = []
    for item in manifest["datasets"]:
        path = data_dir / item["filename"]
        if sha256(path) != item["stored_sha256"]:
            raise ValueError(f"Stored hash mismatch: {path}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["uncompressed_sha256"]:
            raise ValueError(f"Content hash mismatch: {path}")
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            counts = Counter(row["Cyclization"] for row in reader)
        if columns != item["columns"] or sum(counts.values()) != item["rows"] or dict(counts) != item["cyclization_counts"]:
            raise ValueError(f"Schema/count mismatch: {path}")
        if unpack and path.suffix == ".gz":
            target = data_dir / item["uncompressed_filename"]
            if target.exists():
                if sha256(target) != digest:
                    raise ValueError(f"Refusing to replace a different existing file: {target}")
            else:
                temporary = target.with_suffix(target.suffix + ".tmp")
                try:
                    with gzip.open(path, "rb") as src, temporary.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    if sha256(temporary) != digest:
                        raise ValueError(f"Decompressed hash mismatch: {temporary}")
                    temporary.rename(target)
                finally:
                    temporary.unlink(missing_ok=True)
        record = {"filename": item["filename"], "rows": item["rows"], "columns": columns, "cyclization_counts": dict(counts), "content_sha256": digest}
        records.append(record)
        print(f"Verified {path.name}: {item['rows']} rows", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--unpack", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = verify(args.data_dir, args.unpack)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
