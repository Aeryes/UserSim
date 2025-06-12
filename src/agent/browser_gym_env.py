import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException
import numpy as np
import time

from config.constants import broadcast_log
from agent.llm_planner import BedrockLLM


class BrowserGymEnv(gym.Env):
    def __init__(self, use_llm=True, llm=None, start_url="https://example.com", max_obs_tokens=5000):
        super().__init__()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.use_llm = use_llm
        self.llm = llm if use_llm else None
        self.start_url = start_url
        self.max_obs_tokens = max_obs_tokens
        self.action_lookup = []
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.max_obs_tokens,), dtype=np.uint8)

        self.seen_user_stories = set()
        self.visited_urls = set()
        self.visited_dom_elements = set()
        self.last_state = None
        self.last_action = None

    def _get_observation(self):
        try:
            dom = self.driver.execute_script("return document.body.innerHTML")
        except:
            dom = ""
        tokens = self._dom_to_token_list(dom)
        vector = [min(ord(c), 255) for c in tokens[:self.max_obs_tokens]]
        vector += [0] * (self.max_obs_tokens - len(vector))
        return np.array(vector, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.driver.get(self.start_url)
        time.sleep(1)
        self._handle_alerts()
        self.action_lookup = self._get_valid_actions()
        self.seen_user_stories.clear()
        self.visited_urls.clear()
        self.visited_dom_elements.clear()
        self.last_state = None
        self.last_action = None
        self.visited_urls.add(self.driver.current_url)
        obs = self._get_observation()
        return obs, {}

    def step(self, action_idx):
        if action_idx >= len(self.action_lookup):
            return self._get_observation(), -1.0, True, False, {}

        action = self.action_lookup[action_idx]
        self._execute_action(action)
        reward = self._generate_user_story_reward()
        terminated = reward == 1.0
        truncated = False
        self.action_lookup = self._get_valid_actions()
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _dom_to_token_list(self, html):
        html = html.replace("<", " <").replace(">", "> ")
        tokens = html.split()
        return " ".join(tokens)

    def _get_valid_actions(self):
        actions = []
        try:
            for b in self.driver.find_elements(By.TAG_NAME, "button"):
                text = b.text.strip()
                if text:
                    key = f"click_button:{text}"
                    if key not in self.visited_dom_elements:
                        actions.append(key)

            for i in self.driver.find_elements(By.TAG_NAME, "input"):
                name = i.get_attribute("name")
                if name:
                    key = f"type:{name}:test123"
                    if key not in self.visited_dom_elements:
                        actions.append(key)

            for l in self.driver.find_elements(By.TAG_NAME, "a"):
                text = l.text.strip()
                href = l.get_attribute("href")
                if text and href:
                    key = f"click_link:{text}"
                    if key not in self.visited_dom_elements:
                        actions.append(key)
        except Exception as e:
            broadcast_log(f"⚠️ Error while scraping DOM for actions: {e}")

        if self.use_llm and not self.action_lookup:
            try:
                dom = self.driver.page_source
                css = self.driver.execute_script("""
                    return Array.from(document.styleSheets)
                        .map(sheet => {
                            try {
                                return sheet.ownerNode && sheet.ownerNode.outerHTML;
                            } catch (e) {
                                return '';
                            }
                        }).join('\\n');
                """)

                js = self.driver.execute_script("""
                    return Array.from(document.scripts)
                        .map(script => {
                            try {
                                return script.outerHTML;
                            } catch (e) {
                                return '';
                            }
                        }).join('\\n');
                """)

                visited_str = "\n".join(self.visited_urls)
                prompt = (
                    f"Visited URLs so far:\n{visited_str}\n\n"
                    f"Please return a precise, ordered list of next user actions to explore or test this page."
                )
                llm_response = self.llm.query(prompt, dom=dom, css=css, js=js)
                broadcast_log(f"🧠 Raw LLM Response:\n{llm_response}")

                parsed_actions = BedrockLLM.resolve_llm_suggestion(llm_response, valid_actions=actions)
                if parsed_actions:
                    broadcast_log(f"✅ Parsed LLM Actions:\n" + "\n".join(parsed_actions))
                    return parsed_actions[:10]
                else:
                    broadcast_log("⚠️ No valid LLM actions parsed. Falling back.")
            except Exception as e:
                broadcast_log(f"❌ LLM action generation failed: {e}")

        return actions[:10]

    def _handle_alerts(self):
        try:
            alert = self.driver.switch_to.alert
            text = alert.text
            alert.accept()
            broadcast_log(f"⚠️ Dismissed alert: {text}")
        except NoAlertPresentException:
            pass

    def _execute_action(self, action):
        try:
            self.visited_dom_elements.add(action)
            parts = action.split(":", 2)
            if parts[0] == "click_button":
                for b in self.driver.find_elements(By.TAG_NAME, "button"):
                    if b.text.strip() == parts[1]:
                        b.click()
                        break
            elif parts[0] == "click_link":
                for l in self.driver.find_elements(By.TAG_NAME, "a"):
                    if l.text.strip() == parts[1]:
                        l.click()
                        break
            elif parts[0] == "type":
                inputs = self.driver.find_elements(By.NAME, parts[1])
                for inp in inputs:
                    inp.clear()
                    inp.send_keys(parts[2])
        except UnexpectedAlertPresentException as e:
            broadcast_log(f"❌ Alert interrupted action: {str(e)}")
        except Exception as e:
            broadcast_log(f"⚠️ Error while executing action '{action}': {e}")
        finally:
            self._handle_alerts()
            time.sleep(1)
            self.last_action = action
            self.last_state = self.get_state()
            self.visited_urls.add(self.driver.current_url)

    def _generate_user_story_reward(self):
        if not self.use_llm or not self.llm:
            return -0.1

        if not self.last_state or not self.last_action:
            return -0.1

        url = self.last_state["url"]
        dom = self.driver.execute_script("return document.body.innerHTML")
        dom_snippet = dom[:800].strip().replace("```", "")

        prompt = (
            f"Based on the following action and DOM, classify if this path is novel or repeated. Then return:\n"
            f"1. Classification (novel or repeated)\n"
            f"2. A user story\n"
            f"3. A pytest-style Selenium test\n"
            f"4. A snippet of matching DOM\n\n"
            f"URL: {url}\n"
            f"Action: {self.last_action}\n"
            f"DOM:\n{dom_snippet}\n"
        )

        try:
            response = self.llm.query(prompt).strip()
            broadcast_log(f"📘 LLM User Story Classification + Test:\n{response}")

            if response in self.seen_user_stories:
                return -0.1
            else:
                self.seen_user_stories.add(response)
                return 1.0
        except Exception as e:
            broadcast_log(f"⚠️ Failed to generate story/test/dom: {e}")
            return -0.1

    def perform_action(self, action: str):
        return self._execute_action(action)

    def get_state(self):
        url = self.driver.current_url
        buttons = [b.text.strip() for b in self.driver.find_elements(By.TAG_NAME, "button") if b.text.strip()]
        inputs = [i.get_attribute("name") for i in self.driver.find_elements(By.TAG_NAME, "input") if i.get_attribute("name")]
        links = [l.text.strip() for l in self.driver.find_elements(By.TAG_NAME, "a") if l.text.strip()]
        return {"url": url, "buttons": buttons, "inputs": inputs, "links": links}

    def random_action(self) -> list[str]:
        # Basic fallback: randomly click a visible button
        import random
        all_actions = self.action_lookup
        if not all_actions:
            return []
        return [random.choice(all_actions)]

    def check_reward(self):
        return self._generate_user_story_reward()

    def get_dom_context(self) -> tuple[str, str, str]:
        try:
            dom = self.driver.page_source
        except:
            dom = ""
        try:
            css = self.driver.execute_script("""
                return Array.from(document.styleSheets)
                    .map(sheet => {
                        try {
                            return sheet.ownerNode && sheet.ownerNode.outerHTML;
                        } catch (e) {
                            return '';
                        }
                    }).join('\\n');
            """)
        except:
            css = ""
        try:
            js = self.driver.execute_script("""
                return Array.from(document.scripts)
                    .map(script => {
                        try {
                            return script.outerHTML;
                        } catch (e) {
                            return '';
                        }
                    }).join('\\n');
            """)
        except:
            js = ""
        return dom, css, js
