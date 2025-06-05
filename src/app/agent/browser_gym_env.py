import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
import numpy as np
import time

from app.agent.constants import MAX_OBS_TOKENS, broadcast_log
from app.agent.llm_planner import LocalLLM


class BrowserGymEnv(gym.Env):
    def __init__(self, use_llm=True, llm=None):
        super().__init__()
        self.driver = webdriver.Chrome()
        self.use_llm = use_llm
        self.llm = llm if use_llm else None
        self.start_url = "http://127.0.0.1:5000"
        self.action_lookup = []
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(MAX_OBS_TOKENS,), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.driver.get(self.start_url)
        time.sleep(1)
        self.action_lookup = self._get_valid_actions()
        obs = self._get_observation()
        return obs, {}

    def step(self, action_idx):
        if action_idx >= len(self.action_lookup):
            return self._get_observation(), -1.0, True, False, {}

        action = self.action_lookup[action_idx]
        self._execute_action(action)
        reward = self._get_reward()
        terminated = reward == 1.0
        truncated = False
        self.action_lookup = self._get_valid_actions()
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        try:
            dom = self.driver.execute_script("return document.body.innerHTML")
        except:
            dom = ""
        tokens = self._dom_to_token_list(dom)
        vector = [ord(c) for c in tokens[:MAX_OBS_TOKENS]]
        vector += [0] * (MAX_OBS_TOKENS - len(vector))
        return np.array(vector, dtype=np.uint8)

    def _dom_to_token_list(self, html):
        html = html.replace("<", " <").replace(">", "> ")
        tokens = html.split()
        return " ".join(tokens)

    def _get_valid_actions(self):
        actions = []
        try:
            for b in self.driver.find_elements(By.TAG_NAME, "button"):
                if b.text.strip():
                    actions.append(f"click_button:{b.text.strip()}")

            for i in self.driver.find_elements(By.TAG_NAME, "input"):
                name = i.get_attribute("name")
                if name:
                    actions.append(f"type:{name}:test123")

            for l in self.driver.find_elements(By.TAG_NAME, "a"):
                if l.text.strip():
                    actions.append(f"click_link:{l.text.strip()}")

        except Exception as e:
            broadcast_log(f"⚠️ Error while scraping DOM for actions: {e}")

        if self.use_llm:
            dom_summary = self._dom_to_token_list(self.driver.page_source)
            prompt = f"Given the DOM: {dom_summary[:1000]}. What are 10 useful user actions?"

            llm_response = self.llm.query(prompt)
            broadcast_log(f"🧠 Raw LLM Response:\n{llm_response}")

            # Parse LLM response into resolved actions
            parsed_actions = LocalLLM.resolve_llm_suggestion(llm_response, valid_actions=actions)

            if parsed_actions:
                broadcast_log(f"✅ Parsed LLM Actions:\n" + "\n".join(parsed_actions))
                return parsed_actions[:10]

            broadcast_log("⚠️ No parsed actions matched. Falling back to defaults.")

        return actions[:10]

    def _execute_action(self, action):
        try:
            parts = action.split(":", 2)
            if parts[0] == "click_button":
                for b in self.driver.find_elements(By.TAG_NAME, "button"):
                    if b.text.strip() == parts[1]:
                        b.click(); break
            elif parts[0] == "click_link":
                for l in self.driver.find_elements(By.TAG_NAME, "a"):
                    if l.text.strip() == parts[1]:
                        l.click(); break
            elif parts[0] == "type":
                inputs = self.driver.find_elements(By.NAME, parts[1])
                for inp in inputs:
                    inp.clear()
                    inp.send_keys(parts[2])
        except:
            pass
        time.sleep(1)

    def _get_reward(self):
        url = self.driver.current_url
        if "dashboard" in url:
            return 1.0
        elif "signup" in url or "login" in url:
            return 0.1
        return -0.1

    def perform_action(self, action: str):
        return self._execute_action(action)

