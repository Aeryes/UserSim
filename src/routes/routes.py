import queue
from flask import Blueprint, render_template, request, jsonify
from io import StringIO
import sys

import threading
from agent.rl_agent import DumbAgent
from agent.browser_gym_env import BrowserGymEnv
from agent.llm_planner import BedrockLLM
from config.constants import log_subscribers

dashboard = Blueprint("dashboard", __name__)

# Track running agent globally
running_agent = None

@dashboard.route("/", methods=["GET"])
def index():
    return render_template("training_dashboard.html")

@dashboard.route("/run-training", methods=["POST"])
def run_training():
    global running_agent
    try:
        episodes = int(request.form.get("episodes", 10))
        use_llm = request.form.get("use_llm", "true").lower() == "true"
        start_url = request.form.get("start_url", "https://example.com").strip()
        max_obs_tokens = int(request.form.get("max_obs_tokens", 5000))

        llm = BedrockLLM() if use_llm else None
        env = BrowserGymEnv(use_llm=use_llm, llm=llm, start_url=start_url, max_obs_tokens=max_obs_tokens)
        agent = DumbAgent(llm_ins=llm)
        running_agent = agent

        def background_training():
            output_capture = StringIO()
            sys.stdout = output_capture
            try:
                agent.run(env, episodes=episodes, use_llm=use_llm)
            finally:
                sys.stdout = sys.__stdout__
                log_subscribers[0].put("✅ Training complete.")

        threading.Thread(target=background_training).start()

        return render_template("training_dashboard.html", output="🚀 Training started in background...", success=True)

    except Exception as e:
        sys.stdout = sys.__stdout__
        return render_template("training_dashboard.html", error=str(e))


@dashboard.route("/logs")
def get_logs():
    if "log_queue" not in globals():
        global log_queue
        log_queue = queue.Queue()
        log_subscribers.append(log_queue)

    logs = []
    try:
        while True:
            logs.append(log_queue.get_nowait())
    except queue.Empty:
        pass

    return jsonify({"logs": logs})

@dashboard.route("/stop-training", methods=["POST"])
def stop_training():
    global running_agent
    if running_agent:
        running_agent.stop()  # ✅ Now this will work
        return jsonify({"status": "stopping"})
    return jsonify({"status": "no_agent_running"})

@dashboard.route("/banking")
def banking_test_page():
    return render_template("banking_test.html")

@dashboard.route("/v1/wipe-logs", methods=["POST"])
def wipe_logs():
    from config.constants import log_subscribers
    for sub in log_subscribers:
        sub.clear()
    return jsonify(success=True)
