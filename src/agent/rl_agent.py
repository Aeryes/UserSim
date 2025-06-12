from collections import defaultdict
from config.constants import broadcast_log


class DumbAgent:
    def __init__(self, llm_ins=None):
        self.llm = llm_ins
        self.memory = []
        self.should_stop = False
        self.generated_stories = set()
        self.llm_action_memory = defaultdict(list)       # URL -> list of all LLM-suggested actions
        self.completed_actions = defaultdict(set)        # URL -> set of completed actions

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
        self.exploration_tracker = defaultdict(lambda: {
            "buttons": set(),
            "inputs": set(),
            "links": set(),
            "interacted": set()
        })
        self.fully_explored_urls = set()

    def stop(self):
        self.should_stop = True
        broadcast_log("🚑 Stop signal received. Preparing to terminate training...")

    def step(self, env, use_llm=False):
        state = env.get_state()
        url = state["url"]
        self.metrics["state_visits"][url] += 1
        broadcast_log(f"🌐 Visiting URL: {url}")
        broadcast_log(f"🔍 State: buttons={state['buttons']}, inputs={state['inputs']}, links={state['links']}")

        self._update_known_elements(url, state)

        # Get or reuse stored LLM actions
        if url not in self.llm_action_memory or not self.llm_action_memory[url]:
            actions = env.action_lookup
            self.llm_action_memory[url] = actions
            broadcast_log(f"🧠 Stored LLM Actions for {url}: {actions}")

        unexplored_actions = [
            a for a in self.llm_action_memory[url]
            if a not in self.completed_actions[url]
        ]

        if not unexplored_actions:
            broadcast_log(f"✅ All LLM actions explored for {url}. Ending test early.")
            return 0  # Or return a negative reward to indicate no more exploration
        else:
            broadcast_log(f"🚦Next Unexplored Actions: {unexplored_actions}")

        episode_reward = 0
        redirected = False

        for i, selected_action in enumerate(unexplored_actions):
            broadcast_log(f"✅ Executing Action: {selected_action}")
            previous_url = env.driver.current_url

            env.perform_action(selected_action)
            self._track_interaction(url, selected_action)
            self.metrics["action_log"].append(selected_action)

            # ✅ Immediately mark as completed
            self.completed_actions[url].add(selected_action)
            self.llm_action_memory[url] = [a for a in self.llm_action_memory[url] if a != selected_action]

            # Check for redirect
            if env.driver.current_url != previous_url:
                broadcast_log(f"🔄 URL changed to {env.driver.current_url}, ending step.")
                redirected = True
                break

        # ✅ Run LLM reward check only ONCE per episode
        reward = env.check_reward()
        broadcast_log(f"🎯 Final Episode Reward: {reward}")
        episode_reward += reward
        self.memory.append((state, unexplored_actions, reward))

        # Track LLM metrics
        if use_llm:
            self.metrics["llm_successes"] += int(reward > 0)
            key = "LLM_Used_Success" if reward > 0 else "LLM_Used_Fail"
        else:
            key = "NoLLM_Success" if reward > 0 else "NoLLM_Fail"
        self.metrics["llm_confusion"][key] += 1

        self._check_if_url_fully_explored(url, state)
        return episode_reward

    def run(self, env, episodes=10, use_llm=False):
        obs, _ = env.reset()
        first_state = env.get_state()
        first_url = first_state["url"]
        self._update_known_elements(first_url, first_state)
        self.metrics["state_visits"][first_url] += 1

        for ep in range(episodes):
            if self.should_stop:
                broadcast_log("⏹️ Training stopped by user.")
                break

            all_done = all(
                self._url_fully_explored(url, self.exploration_tracker[url])
                for url in self.exploration_tracker
            )
            if all_done:
                broadcast_log("✅ All known URLs and DOM elements have been explored. Ending training.")
                break

            broadcast_log(f"🚀 Starting Episode {ep + 1}")
            total_reward = 0
            steps = 0

            for step_num in range(50):
                if self.should_stop:
                    broadcast_log("⏹️ Training interrupted mid-episode.")
                    return

                broadcast_log(f"📌 Step {step_num + 1}")
                r = self.step(env, use_llm=use_llm)
                total_reward += r
                steps += 1

            self.metrics["total_rewards"].append(total_reward)
            if total_reward > 0:
                self.metrics["success_count"] += 1
                self.metrics["steps_to_goal"].append(steps)
                broadcast_log("🏆 Goal Achieved!")
            broadcast_log(f"📊 Total Reward for Episode {ep + 1}: {total_reward}")
        self.print_summary()

    def _track_interaction(self, url, action):
        if action.startswith("click_button:"):
            label = action.split("click_button:")[1]
            self.exploration_tracker[url]["buttons"].add(label)
            self.exploration_tracker[url]["interacted"].add(f"button:{label}")
        elif action.startswith("type:"):
            parts = action.split(":")
            if len(parts) == 3:
                self.exploration_tracker[url]["inputs"].add(parts[1])
                self.exploration_tracker[url]["interacted"].add(f"input:{parts[1]}")
        elif action.startswith("click_link:"):
            label = action.split("click_link:")[1]
            self.exploration_tracker[url]["links"].add(label)
            self.exploration_tracker[url]["interacted"].add(f"link:{label}")

    def _update_known_elements(self, url, state):
        self.exploration_tracker[url]["buttons"].update(state["buttons"])
        self.exploration_tracker[url]["inputs"].update(state["inputs"])
        self.exploration_tracker[url]["links"].update(state["links"])

    def _check_if_url_fully_explored(self, url, state):
        if self._url_fully_explored(url, self.exploration_tracker[url]):
            if url not in self.fully_explored_urls:
                self.fully_explored_urls.add(url)
                broadcast_log(f"✅ URL fully explored: {url}")

    def _url_fully_explored(self, url, tracker):
        all_buttons = all(f"button:{b}" in tracker["interacted"] for b in tracker["buttons"])
        all_inputs = all(f"input:{i}" in tracker["interacted"] for i in tracker["inputs"])
        all_links = all(f"link:{l}" in tracker["interacted"] for l in tracker["links"])
        return all_buttons and all_inputs and all_links

    def _check_user_story(self, state, action):
        url = state["url"]
        if "dashboard" in url and "access_dashboard" not in self.generated_stories:
            self.generated_stories.add("access_dashboard")
            prompt = (
                "Generate a user story for this event: "
                "Include the purpose of the story, matching DOM snippet, and a pytest function to test it."
            )
            story = self.llm.query(prompt)[0]
            broadcast_log("📘 LLM User Story Classification + Test:")
            broadcast_log(story)

    def print_summary(self):
        print("\n📊 Summary Metrics:")
        num_episodes = len(self.metrics["total_rewards"])
        if num_episodes == 0:
            print("⚠️ No episodes were completed.")
            return

        print(f"✅ Success Rate: {self.metrics['success_count']} / {num_episodes}")
        avg_steps = sum(self.metrics["steps_to_goal"]) / len(self.metrics["steps_to_goal"]) if self.metrics["steps_to_goal"] else "N/A"
        avg_reward = sum(self.metrics["total_rewards"]) / num_episodes
        print(f"🏃‍♂️ Avg Steps to Goal: {avg_steps}")
        print(f"🌟 Avg Total Reward: {avg_reward}")
        print(f"🧠 LLM Prompted: {self.metrics['llm_prompts']} | LLM Helped: {self.metrics['llm_successes']}")
        action_set = set(self.metrics["action_log"])
        print(f"🔁 Action Diversity: {len(action_set)} unique actions")
