import os, boto3, pandas as pd, sklearn
from dotenv import load_dotenv

load_dotenv()
bucket, _, prefix = os.environ["S3_PATH"].removeprefix("s3://").partition("/")
s3 = boto3.Session(profile_name=os.getenv("AWS_PROFILE")).client("s3")
resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)

print("pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("S3 objects visible:", resp.get("KeyCount"))

