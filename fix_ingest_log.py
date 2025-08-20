# fix_ingest_log.py
import csv, pathlib

p = pathlib.Path("logs/ingest_log.csv")
if not p.exists():
    print("No logs/ingest_log.csv found.")
    raise SystemExit

expected = [
    "ts_utc","namespace","type","title","vector_id",
    "content_hash","chars","model","status","error",
    "source","filename"
]

rows = []
with p.open("r", encoding="utf-8", newline="") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        # pad or trim to 12 columns
        if len(row) < len(expected):
            row += [""] * (len(expected) - len(row))
        elif len(row) > len(expected):
            row = row[:len(expected)]
        rows.append(row)

with p.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(expected)
    w.writerows(rows)

print("Normalized logs/ingest_log.csv to 12 columns.")
