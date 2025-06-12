import os
import boto3
from datetime import datetime

MAX_OBS_TOKENS = 5000
START_URL = ""

log_subscribers = []

def broadcast_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message.strip()}\n"
    for q in log_subscribers:
        q.put(formatted)

# Determine environment profile
ENV = os.getenv("APP_ENV", "local")  # local or dev

# AWS config per environment
AWS_REGION = "us-east-1"
AWS_PROFILE = "bedrock-local" if ENV == "local" else None

def get_bedrock_client():
    """Returns a boto3 client using the appropriate profile or IAM role."""
    if AWS_PROFILE:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    else:
        session = boto3.Session(region_name=AWS_REGION)
    return session.client("bedrock-runtime")
