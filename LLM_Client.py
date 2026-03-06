from langchain_ollama import ChatOllama
import pandas as pd
import json
import time
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment variables


class LLM_Client:
    def __init__(self, logging, temperature=0., model='gpt-oss:20b', max_tokens=4096):
        self.logging = logging
        self.temperature = temperature
        self.model = model

        self.max_tokens = max_tokens

        self.total_completion_tokens_usage = 0
        self.total_prompt_tokens_usage = 0

        # Initialize Ollama client
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        api_key = os.getenv("OLLAMA_API_KEY", None)

        # Create ChatOllama with proper headers for API key authentication
        client_kwargs = {}
        if api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

        self.ollama_client = ChatOllama(
            model=model,
            base_url=base_url,
            client_kwargs=client_kwargs,
            temperature=temperature,
            num_predict=max_tokens,
            format='json'
        )

    def get_chat_completion(self, messages):
        try:
            response = self.ollama_client.invoke(messages)
            res_content = response.content
            # Ollama doesn't provide token usage by default, return 0
            completion_tokens_usage = 0
            prompt_tokens_usage = 0

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content
        except Exception as e:
            self.logging.error(f'Ollama API call failed: {e}')
            raise


    def load_json_response(self, res_content):
        try:  # 带Markdown code block notation e.g., '\n```json\n{\n  "Choice": "ACDB",\n
              # "Reason": "The travel time for ACDB was the shortest yesterday."\n}\n```'
            json_str = res_content.strip()[7:-3].strip()
            data = json.loads(json_str)
            route, reason = data["Choice"].lower(), data["Reason"]
            return data
        except:
            try:
                data = json.loads(res_content)
                route, reason = data["Choice"].lower(), data["Reason"]
                return data
            except:
                try:
                    json_str = res_content.strip()[3:-3].strip()
                    data = json.loads(json_str)
                    route, reason = data["Choice"].lower(), data["Reason"]
                    return data

                except:
                    print('Not Json response!!!!!!!!!!!!!!!!!!!!!!')
                    raise ValueError('Not Json response!')


    def get_chat_completion_with_retries(self, messages, max_retries=5):
        success = False
        retry = 0
        max_retries = max_retries
        while retry < max_retries and not success:
            try:
                response = self.get_chat_completion(messages=messages)
                json_res = self.load_json_response(response)
                success = True
            except Exception as e:
                #print(f"Error: {e}\ nRetrying ...")
                self.logging.error(f'traveller get_chat_completion from llm failed !')
                retry += 1
                time.sleep(5)

        return json_res  # {'Choice': 'ACDB', 'Reason': 'The travel time for ACDB was the shortest yesterday.'}

    def calculate_cost(self):
        cost_info = {'total_prompt_tokens_usage': 0,
                     'total_completion_tokens_usage': 0,
                     'input_unit_cost': 0, 'output_unit_cost': 0,
                     'input_cost': 0, 'completion_cost': 0,
                     'total_cost': 0, 'unit': 'Yuan'}

        return cost_info
