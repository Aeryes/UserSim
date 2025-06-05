import os
import json

# Config file paths
CONFIG_DIR = os.path.join("config")
GENERAL_CONFIG = os.path.join(CONFIG_DIR, "general_config.json")
MODEL_CONFIG = os.path.join(CONFIG_DIR, "model_config.json")
DOWNLOAD_STATUS = os.path.join(CONFIG_DIR, "download_status.json")

# Create config directory if it doesn't exist
os.makedirs(CONFIG_DIR, exist_ok=True)

# Initialize general_config.json
if not os.path.exists(GENERAL_CONFIG) or os.path.getsize(GENERAL_CONFIG) == 0:
    with open(GENERAL_CONFIG, "w") as f:
        json.dump({"model_storage_path": "G:/ML Models"}, f, indent=2)
    print(f"✅ Created {GENERAL_CONFIG}")

# Initialize model_config.json
if not os.path.exists(MODEL_CONFIG) or os.path.getsize(MODEL_CONFIG) == 0:
    with open(MODEL_CONFIG, "w") as f:
        json.dump([], f, indent=2)
    print(f"✅ Created {MODEL_CONFIG}")

# Initialize download_status.json
if not os.path.exists(DOWNLOAD_STATUS) or os.path.getsize(DOWNLOAD_STATUS) == 0:
    with open(DOWNLOAD_STATUS, "w") as f:
        json.dump({}, f, indent=2)
    print(f"✅ Created {DOWNLOAD_STATUS}")
