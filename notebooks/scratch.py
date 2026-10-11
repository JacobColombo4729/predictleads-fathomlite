import os, boto3, pandas as pd, sklearn
from dotenv import load_dotenv

load_dotenv()
s3_path = os.getenv("S3_PATH")
if not s3_path:
    raise RuntimeError("S3_PATH is not set — add `S3_PATH=s3://<bucket>/<prefix>/` on its own line in .env")
BUCKET, _, PREFIX = s3_path.strip().removeprefix("s3://").partition("/")

s3 = boto3.Session(profile_name=os.getenv("AWS_PROFILE")).client("s3")
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=5)

print("pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("S3 objects visible:", resp.get("KeyCount"))

