import difflib
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    Gemma3ForCausalLM, BitsAndBytesConfig  # <- specifically for Gemma 3 models
)
import torch
import os
import json

from typing import List, Optional

from app.agent.constants import broadcast_log


def is_local_model(path):
    return os.path.exists(path) and os.path.isdir(path)

def get_model_type(model_path):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("model_type", "").lower()
    return ""

class LocalLLM:
    def __init__(self, model_path=None):
        self.tokenizer = None
        self.model = None
        self.is_gemma = False

        if not model_path:
            print("ℹ️ No model path provided — skipping LLM loading.")
            return

        try:
            model_path = Path(model_path)
            model_name = model_path.name.lower()

            config_path = model_path / "config.json"
            model_type = ""
            if config_path.exists():
                with open(config_path, "r") as f:
                    model_type = json.load(f).get("model_type", "").lower()

            print(f"🔍 Detected model type: {model_type or model_name}")

            if "gemma" in model_type or "gemma" in model_name:
                print("🧬 Model detected as: Gemma")
                self.is_gemma = True
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

                self.model = Gemma3ForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quant_config
                ).eval()

                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            elif "llama" in model_type or "llama" in model_name:
                print("🦙 Model detected as: LLaMA")
                tokenizer_file = model_path / "tokenizer.model"
                if tokenizer_file.exists():
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        use_fast=False,
                        legacy=True,
                        tokenizer_file=str(tokenizer_file)
                    )
                else:
                    print(f"⚠️ No tokenizer.model found at: {tokenizer_file}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        use_fast=False,
                        legacy=True
                    )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            else:
                print("📦 Model detected as: General/Other")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            print("✅ LLM successfully loaded.")

        except Exception as e:
            print(f"❌ Failed to load LLM from {model_path}: {e}")

    def query(self, prompt, max_new_tokens=100):
        if not self.tokenizer or not self.model:
            print("⚠️ LLM not initialized. Skipping inference.")
            return ""

        broadcast_log(f"📤 Prompt to LLM: {prompt}")

        if self.is_gemma:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
            ]

            inputs = self.tokenizer.apply_chat_template(
                [messages],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Only convert float tensors to bfloat16
            for k, v in inputs.items():
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(torch.bfloat16)

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = decoded.replace(prompt, "").strip()

        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response


    def resolve_llm_suggestion(response: str, valid_actions: List[str]) -> Optional[str]:
        """
        Attempts to resolve an LLM suggestion into one of the valid environment actions.
        Uses fuzzy and partial matching to interpret ambiguous LLM responses.

        Args:
            response (str): Raw response from the LLM.
            valid_actions (List[str]): List of valid action strings in the environment.

        Returns:
            Optional[str]: The best-matched action, or None if no suitable match found.
        """
        # Normalize response
        response = response.strip().lower()

        # Direct substring match
        for action in valid_actions:
            if response in action.lower() or action.lower() in response:
                return action

        # Fuzzy match using difflib
        best_match = difflib.get_close_matches(response, valid_actions, n=1, cutoff=0.3)
        if best_match:
            return best_match[0]

        # Partial keyword matching
        for action in valid_actions:
            action_keywords = action.lower().replace('_', ' ').split()
            if any(word in response for word in action_keywords):
                return action

        return None


