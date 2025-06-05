import json
import os
from collections import defaultdict

from app.agent.constants import broadcast_log
from app.agent.llm_planner import LocalLLM


class DumbAgent:
    def __init__(self, llm_ins=None):
        self.llm = llm_ins
        self.memory = []
        self.metrics = {
            "total_rewards": [],
            "success_count": 0,
            "steps_to_goal": [],
            "action_log": [],
            "llm_prompts": 0,
            "llm_successes": 0,
            "state_visits": defaultdict(int),
            "llm_confusion": {
                "LLM_Used_Success": 0,
                "LLM_Used_Fail": 0,
                "NoLLM_Success": 0,
                "NoLLM_Fail": 0,
            }
        }

    def step(self, env, use_llm=False):
        state = env.get_state()
        url = state["url"]
        self.metrics["state_visits"][url] += 1

        actions = []
        if use_llm:
            prompt = f"You are a user trying to understand this page. The URL is {url}. It has buttons: {state['buttons']}, inputs: {state['inputs']}, and links: {state['links']}. What should you do next?"
            suggestion = self.llm.query(prompt).lower()
            self.metrics["llm_prompts"] += 1
            broadcast_log(f"🧠 LLM to Agent Response:\n{suggestion}")

            # Match LLM suggestion to available actions
            for a in env.action_lookup:
                if any(term in a.lower() for term in suggestion.split()):
                    actions.append(a)

        if not actions:
            actions = env.random_action()

        # Apply the first suggested or fallback action
        env.perform_action(actions[0])

        reward = env.check_reward()

        broadcast_log(f"🎬 Actions Taken: {actions} | Reward: {reward}")

        self.memory.append((state, actions, reward))
        self.metrics["action_log"].extend(actions)

        # Reward success/failure metrics...
        if use_llm:
            if reward > 0:
                self.metrics["llm_successes"] += 1
                self.metrics["llm_confusion"]["LLM_Used_Success"] += 1
            else:
                self.metrics["llm_confusion"]["LLM_Used_Fail"] += 1
        else:
            if reward > 0:
                self.metrics["llm_confusion"]["NoLLM_Success"] += 1
            else:
                self.metrics["llm_confusion"]["NoLLM_Fail"] += 1

        return reward

    def run(self, env, episodes=10, use_llm=False):
        for ep in range(episodes):
            print(f"▶️ Episode {ep+1}")
            total_reward = 0
            env.reset()
            steps = 0
            for _ in range(20):
                r = self.step(env, use_llm=use_llm)
                steps += 1
                total_reward += r
                if r == 1.0:
                    self.metrics["success_count"] += 1
                    self.metrics["steps_to_goal"].append(steps)
                    break
            self.metrics["total_rewards"].append(total_reward)
            print(f"Total Reward: {total_reward}")
        self.print_summary()

    def print_summary(self):
        print("\n📊 Summary Metrics:")
        print(f"✅ Success Rate: {self.metrics['success_count']} / {len(self.metrics['total_rewards'])}")
        print(f"🏃‍♂️ Avg Steps to Goal: {sum(self.metrics['steps_to_goal']) / len(self.metrics['steps_to_goal']) if self.metrics['steps_to_goal'] else 'N/A'}")
        print(f"🎯 Avg Total Reward: {sum(self.metrics['total_rewards']) / len(self.metrics['total_rewards'])}")
        print(f"🧠 LLM Prompted: {self.metrics['llm_prompts']} | LLM Helped: {self.metrics['llm_successes']}")
        action_set = set(self.metrics["action_log"])
        print(f"🔁 Action Diversity: {len(action_set)} unique actions")
