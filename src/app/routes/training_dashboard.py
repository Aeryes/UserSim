import os
import io
import json
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Blueprint, render_template, request, redirect, url_for, send_file, Response, stream_with_context

from stable_baselines3 import PPO
from app.agent.browser_gym_env import BrowserGymEnv
from app.agent.constants import DOWNLOAD_STATUS_FILE, MODEL_CONFIG_PATH, get_general_config, log_subscribers
from app.agent.llm_planner import LocalLLM
from app.agent.rl_agent import DumbAgent
from huggingface_hub import snapshot_download
from app.utilities.CustomTqdm import CustomTqdm

admin_bp = Blueprint("admin", __name__, url_prefix="/v1")
latest_agent = None  # 🧠 Track last trained DumbAgent instance


class TqdmWithCallback(CustomTqdm):
    def __init__(self, *args, **kwargs):
        callback = kwargs.pop("callback", None)
        super().__init__(*args, callback=callback, **kwargs)


def update_download_status(model_id, status=None, progress=None, error=None):
    try:
        with open(DOWNLOAD_STATUS_FILE, "r") as f:
            content = f.read().strip()
            data = json.loads(content) if content else {}
    except FileNotFoundError:
        data = {}

    if model_id not in data:
        data[model_id] = {}

    if status is not None:
        data[model_id]["status"] = status
    if progress is not None:
        data[model_id]["progress"] = progress
    if error is not None:
        data[model_id]["error"] = error

    with open(DOWNLOAD_STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_download_status(model_id):
    try:
        with open(DOWNLOAD_STATUS_FILE, "r") as f:
            data = json.load(f)
        return data.get(model_id, {})
    except:
        return {}


def load_model_config(config_path=MODEL_CONFIG_PATH):
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []


def save_model_config(config, config_path=MODEL_CONFIG_PATH):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


@admin_bp.route("/training-dashboard", methods=["GET"])
def training_dashboard():
    models = load_model_config()
    metrics = latest_agent.metrics if latest_agent else {}
    return render_template("training_dashboard.html", metrics=metrics, models=models)


@admin_bp.route("/training-dashboard/train", methods=["POST"])
def trigger_training():
    global latest_agent

    model_path = request.form.get("model_path", "").strip()
    model_path = os.path.abspath(os.path.normpath(model_path))

    lr = float(request.form.get("lr", 0.0003))
    steps = int(request.form.get("steps", 10000))
    batch_size = int(request.form.get("batch_size", 64))

    if not os.path.exists(model_path):
        models = load_model_config()
        error_message = f"⚠️ Model path does not exist: {model_path}"
        print(error_message)
        return render_template("training_dashboard.html", metrics={}, models=models,
                               error_message=error_message)

    llm_instance = LocalLLM(model_path)
    latest_agent = DumbAgent(llm_instance)  # Uses same path internally
    env = BrowserGymEnv(use_llm=False, llm=llm_instance)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr, batch_size=batch_size, device="cpu")
    model.learn(total_timesteps=steps)
    model.save("ppo_browser_agent")

    latest_agent.run(env, episodes=10, use_llm=True)


    return render_template("training_dashboard.html", metrics=latest_agent.metrics, models=load_model_config())



@admin_bp.route("/training-dashboard/download-model", methods=["POST"])
def download_model():
    data = request.get_json()
    model_id = data.get("hf_model_id")
    display_name = data.get("display_name", model_id.split("/")[-1])

    folder_name = model_id.replace("/", "-")
    general_config = get_general_config()
    model_path = os.path.join(general_config["model_storage_path"], folder_name)

    try:
        update_download_status(model_id, status="downloading", progress=0)
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        update_download_status(model_id, status="complete", progress=100)

        config = load_model_config()
        if not any(entry["path"] == model_path for entry in config):
            config.append({"name": display_name, "path": model_path})
            save_model_config(config)

    except Exception as e:
        update_download_status(model_id, status="error", error=str(e))

    return redirect(url_for("admin.training_dashboard"))


@admin_bp.route("/dashboard/heatmap.png")
def heatmap_image():
    if not latest_agent:
        return "No data", 404
    fig, ax = plt.subplots(figsize=(8, 3))
    visits = latest_agent.metrics["state_visits"]
    ax.bar(visits.keys(), visits.values())
    ax.set_title("Page Visit Frequency Heatmap")
    ax.set_xticklabels(visits.keys(), rotation=45, ha="right")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@admin_bp.route("/dashboard/confusion_image.png")
def confusion_image():
    if not latest_agent:
        return "No data", 404
    cm = latest_agent.metrics["llm_confusion"]
    matrix = np.array([
        [cm["LLM_Used_Success"], cm["LLM_Used_Fail"]],
        [cm["NoLLM_Success"], cm["NoLLM_Fail"]],
    ])
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Success", "Fail"],
                yticklabels=["LLM Used", "No LLM"], cmap="Blues", ax=ax)
    ax.set_title("LLM Confusion Matrix")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@admin_bp.route("/training-dashboard/stream")
def stream():
    return stream_logs()

def stream_logs():
    def event_stream():
        q = queue.Queue()
        log_subscribers.append(q)
        try:
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        except GeneratorExit:
            log_subscribers.remove(q)
    return Response(event_stream(), mimetype="text/event-stream")