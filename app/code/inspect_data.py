import io
import json

import boto3
import zstandard as zstd

from prepare_jobs import BUCKET, PREFIX, s3_session  # S3_PATH and AWS_PROFILE from the top-level .env

session = s3_session()
s3 = session.client("s3")

files = [
    "companies_2026_09_30.jsonl.zst",
    "job_openings_2026_09_30.jsonl.zst",
]

for filename in files:
    print(f"\nInspecting: {filename}")
    response = s3.get_object(Bucket=BUCKET, Key=PREFIX + filename)
    body = response["Body"]

    try:
        with zstd.ZstdDecompressor().stream_reader(body) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for index in range(5):
                    line = text.readline()
                    if not line:
                        break

                    record = json.loads(line)
                    print(f"Record {index + 1}:")

                    if isinstance(record, dict):
                        for field, value in record.items():
                            print(f"  {field}: {type(value).__name__}")

                            if isinstance(value, list):
                                print(f"    Items: {len(value)}")

                                if value and isinstance(value[0], dict):
                                    for key, item in value[0].items():
                                        print(
                                            f"    {key}: "
                                            f"{type(item).__name__}"
                                        )

                                        if isinstance(item, dict):
                                            for nested_key, nested_value in item.items():
                                                print(
                                                    f"      {nested_key}: "
                                                    f"{type(nested_value).__name__}"
                                                )
                    else:
                        print(f"  Record type: {type(record).__name__}")
    finally:
        body.close()