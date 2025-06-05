import json
from datetime import datetime

import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

MAX_OBS_TOKENS = 5000
MODEL_CONFIG_PATH = "config/model_config.json"

def get_general_config():
    with open("config/general_config.json", "r") as f:
        return json.load(f)

MODEL_STORAGE_PATH = get_general_config().get("model_storage_path", "./local_models")

DOWNLOAD_STATUS_FILE = "config/download_status.json"

log_subscribers = []

def broadcast_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message.strip()}\n"
    for q in log_subscribers:
        q.put(formatted)