import json
import time

import boto3

from config.constants import get_bedrock_client


class BedrockLLM:
    def __init__(self, model_id="anthropic.claude-v2"):
        self.model_id = model_id
        self.client = get_bedrock_client()

    def query(self, prompt: str, dom: str = "", css: str = "", js: str = "") -> str:
        time.sleep(2)  # ⏳ Throttle to prevent Bedrock API limits

        # Add DOM, CSS, and JS context for more accurate suggestions
        full_prompt = (
            f"You are a web automation assistant. Your job is to examine a webpage's DOM, CSS, and JS, and suggest "
            f"precise, sequential user actions to explore or test the page. Respond only with a list of exact actions "
            f"(like 'type:username:myuser', 'type:password:123456', 'click_button:Login') in order of execution.\n\n"
            f"--- DOM ---\n{dom[:3000]}\n\n"
            f"--- CSS ---\n{css[:1500]}\n\n"
            f"--- JS ---\n{js[:1500]}\n\n"
            f"--- Prompt ---\n{prompt}\n"
        )

        body = {
            "prompt": f"\n\nHuman: {full_prompt}\n\nAssistant:",
            "max_tokens_to_sample": 300,
            "temperature": 0.7,
            "top_k": 250,
            "top_p": 0.99,
            "stop_sequences": ["\n\nHuman:"],
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["completion"]

    # @staticmethod
    # def resolve_llm_suggestion(text: str, valid_actions: list[str]) -> list[str]:
    #     selected = []
    #     text = text.lower().strip()
    #     lines = [l.strip() for l in text.split("\n") if l.strip()]
    #     valid_action_set = set(a.lower() for a in valid_actions)
    #
    #     for line in lines:
    #         line = line.strip()
    #         if line in valid_action_set:
    #             selected.append(line)
    #         else:
    #             # Allow partial matching for typed inputs
    #             if line.startswith("type:"):
    #                 parts = line.split(":")
    #                 if len(parts) == 3:
    #                     action_prefix = f"type:{parts[1]}:"
    #                     # Replace value to match a valid action prefix
    #                     for a in valid_actions:
    #                         if a.lower().startswith(action_prefix):
    #                             selected.append(a)
    #                             break
    #     return selected[:10]
    @staticmethod
    def resolve_llm_suggestion(text: str, valid_actions: list[str]) -> list[str]:
        selected = []
        text = text.lower().strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:
            if line.startswith("type:") and line.count(":") == 2:
                selected.append(line)
            elif line.startswith("click_button:") or line.startswith("click_link:"):
                selected.append(line)
            # You can add other rules like `select:dropdown:value`, etc.

        return selected[:20]  # allow longer plans

