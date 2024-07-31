import os
import time
from typing import List, Dict, Tuple

import ipdb
import openai
import re

from vllm import LLM, SamplingParams
from wrapt_timeout_decorator import *

from models.openai_prompts import *


def set_openai_api_key():
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        api_key = open(f"/net/nfs.cirrascale/mosaic/jaehunj/OPENAI_API_KEY", "r").read().strip()
        os.environ["OPENAI_API_KEY"] = api_key

    openai.api_key = api_key
    return api_key


class OpenAIModel:
    def __init__(self, model_name: str, dataset_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.client = openai.Client(api_key=set_openai_api_key())

        if self.model_name in ["gpt-3.5-turbo-instruct"]:
            self.use_completion = True  # use completion API
        else:
            self.use_completion = False  # use Chat API

    def generate_y(self, questions: List[str], num_generations: int) -> List[List[str]]:
        prompts = [self.prepare_prompt_for_y(question) for question in questions]

        if any(name in self.dataset_name for name in ["strategy_qa"]):
            prompt_config = {
                "model": self.model_name,
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 500,
                "n": num_generations,
            }
        else:
            raise NotImplementedError

        generations = [self.prompt_generation(prompt, prompt_config) for prompt in prompts]
        return generations

    def prepare_prompt_for_y(self, question: str) -> str:
        question = question.strip()

        if any(name in self.dataset_name for name in ["strategy_qa"]):
            instruction = strategy_qa_instruction
            prompt = f"{instruction}\n" \
                     f"\n" \
                     f"Q: {question}\n" \
                     f"A:"
        else:
            raise NotImplementedError

        return prompt

    @timeout(15, timeout_exception=StopIteration)
    def _request_generation(self, prompt: str, prompt_config: Dict) -> List[str]:
        generations = []

        if self.use_completion:
            # GPT-3.5
            completion = self.client.completions.create(
                prompt=prompt,
                **prompt_config,
            )

            for choice in completion.choices:
                generations.append(choice.text.strip())
        else:
            # ChatGPT
            completion = self.client.chat.completions.create(
                **prompt_config,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            for choice in completion.choices:
                generations.append(choice.message.content.strip())

        return generations

    def prompt_generation(self, prompt: str, prompt_config: Dict) -> List[str]:
        try:
            inferred_answer = self._request_generation(prompt, prompt_config)
            return inferred_answer

        except StopIteration as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(prompt, prompt_config)

        except openai.OpenAIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(prompt, prompt_config)

        except OSError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(prompt, prompt_config)


if __name__ == "__main__":
    questions = [
        "Are more people today related to Genghis Khan than Julius Caesar?",
        "Are more people today related to Genghis Khan than Julius Caesar?"
    ]
    model = OpenAIModel(model_name="gpt-3.5-turbo", dataset_name="strategy_qa")

    batch_generations = model.generate_y(questions, 1)
    ipdb.set_trace()
    pass

