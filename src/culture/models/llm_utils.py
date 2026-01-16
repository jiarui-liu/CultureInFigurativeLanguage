import os
from openai import OpenAI, AsyncOpenAI
from together import Together, AsyncTogether
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError
)
import logging

logger = logging.getLogger(__name__)

class ChatModel:
    """Wrapper for interacting with both OpenAI and Hugging Face models."""

    # Class-level cache to store instances
    _instances = {}

    def __new__(cls, model="gpt-4o-mini", provider="openai", cache_dir=None):
        """Ensures only one instance per model is created."""
        if model in cls._instances:
            return cls._instances[model]

        # Create a new instance if not already cached
        instance = super(ChatModel, cls).__new__(cls)
        cls._instances[model] = instance
        return instance

    def __init__(self, model="gpt-4o-mini", provider="openai", cache_dir=None):
        # Avoid re-initialization if already created
        if hasattr(self, "initialized"):
            return
        self.initialized = True  # Mark as initialized

        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE") or None,
            )
            # Initialize async client
            self.async_client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE") or None,
            )
        elif provider == "togetherai":
            self.client = Together(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                # base_url="https://api.together.xyz/v1",
            )
            # Initialize async client
            self.async_client = AsyncTogether(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                # base_url="https://api.together.xyz/v1",
            )
        elif provider == "bedrock":
            import boto3
            from botocore.exceptions import ClientError
            self.region = os.environ.get("AWS_REGION", "us-east-1")
            self.client = boto3.client("bedrock-runtime", region_name=self.region)
            # For bedrock, we use the same client for both sync and async operations
            self.async_client = self.client
        elif provider == "huggingface":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            if any(
                item in model.lower()
                for item in [
                    "llama-3.1",
                    "qwen-2.5",
                    "mistral",
                ]
            ):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model, cache_dir=cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self.pipeline = pipeline(
                    "text-generation", model=self.model, tokenizer=self.tokenizer
                )
            elif any(
                item in model.lower()
                for item in [
                    "deepseek-r1-distill",
                    "qwq",
                ]
            ):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model, cache_dir=cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    cache_dir=cache_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

        else:
            raise ValueError("Unsupported provider. Use 'openai', 'togetherai', 'bedrock', or 'huggingface'.")

    def generate(self, messages, **gen_kwargs):
        """Send a prompt to the chosen model and return the response."""
        if self.provider in ["openai", "togetherai"]:
            print(messages)
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **gen_kwargs
            )
            return response.choices[0].message.content
        elif self.provider == "bedrock":
            import boto3
            from botocore.exceptions import ClientError
            print(messages)
            # Convert messages to bedrock format
            bedrock_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    bedrock_messages.append({
                        "role": "assistant",
                        "content": [{"text": msg["content"]}]
                    })
                elif msg["role"] == "user":
                    bedrock_messages.append({
                        "role": "user",
                        "content": [{"text": msg["content"]}]
                    })

            try:
                response = self.client.converse(
                    modelId=self.model,
                    messages=bedrock_messages,
                    inferenceConfig=gen_kwargs
                )
                return response['output']['message']['content'][0]['text']
            except (ClientError, Exception) as e:
                print(f"ERROR: Can't invoke '{self.model}'. Reason: {e}")
                raise e
        elif self.provider == "huggingface":
            print(messages)
            if hasattr(self, "pipeline"):
                result = self.pipeline(messages, **gen_kwargs)
                return result[0]["generated_text"][-1]
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(
                    self.model.device
                )
                generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                return response
        else:
            raise ValueError("Invalid provider. Use 'openai', 'togetherai', 'bedrock', or 'huggingface'.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def async_generate(self, messages, **gen_kwargs):
        """Asynchronously send a request to the model and return the response."""
        try:
            if self.provider in ["openai", "togetherai"]:
                response = await self.async_client.chat.completions.create(
                    model=self.model, messages=messages, **gen_kwargs
                )
                return response.choices[0].message.content
            elif self.provider == "bedrock":
                import boto3
                from botocore.exceptions import ClientError
                # Convert messages to bedrock format
                bedrock_messages = []
                for msg in messages:
                    if msg["role"] == "assistant":
                        bedrock_messages.append({
                            "role": "assistant",
                            "content": [{"text": msg["content"]}]
                        })
                    elif msg["role"] == "user":
                        bedrock_messages.append({
                            "role": "user",
                            "content": [{"text": msg["content"]}]
                        })

                try:
                    # Use streaming for async generation
                    streaming_response = self.client.converse_stream(
                        modelId=self.model,
                        messages=bedrock_messages,
                        inferenceConfig=gen_kwargs
                    )

                    # Collect the streamed response
                    response_text = ""
                    for chunk in streaming_response["stream"]:
                        if "contentBlockDelta" in chunk:
                            text = chunk["contentBlockDelta"]["delta"]["text"]
                            response_text += text

                    return response_text
                except (ClientError, Exception) as e:
                    print(f"ERROR: Can't invoke '{self.model}'. Reason: {e}")
                    raise e
            else:
                # For non-OpenAI models, fall back to synchronous method
                return self.generate(messages, **gen_kwargs)
        except Exception as e:
            logger.warning(f"Error in async_generate: {e}")
            raise e

    async def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **gen_kwargs) -> List[str]:
        """Batch process multiple requests asynchronously and return a list of results. Mainly used for single-agent mode."""
        if self.provider not in ["openai", "togetherai", "bedrock"]:
            # For non-async supported models, process each request synchronously
            return [self.generate(messages, **gen_kwargs) for messages in batch_messages]

        tasks = [
            self.async_generate(messages, **gen_kwargs)
            for messages in batch_messages
        ]
        return await asyncio.gather(*tasks)

    def batch_generate_sync(self, batch_messages: List[List[Dict[str, str]]], batch_size=10, **gen_kwargs) -> List[str]:
        """Synchronous batch processing method, using asyncio to run batch tasks. Mainly used for single-agent mode."""
        if self.provider not in ["openai", "togetherai", "bedrock"]:
            # For non-async supported models, process each request synchronously
            return [self.generate(messages, **gen_kwargs) for messages in batch_messages]

        # Process in batches to avoid exceeding API limits
        results = []
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i+batch_size]
            batch_results = asyncio.run(self.batch_generate(batch, **gen_kwargs))
            results.extend(batch_results)
        return results

    async def batch_generate_with_indices(
        self,
        batch_messages: List[Tuple[Any, List[Dict[str, str]]]],
        **gen_kwargs
    ) -> List[Tuple[Any, Optional[str], Optional[Exception]]]:
        """
        Batch process multiple requests asynchronously, preserving indices/metadata.

        Args:
            batch_messages: List of tuples (index/metadata, messages)
            **gen_kwargs: Generation kwargs passed to the model

        Returns:
            List of tuples (index/metadata, response_or_None, exception_or_None)
        """
        if self.provider not in ["openai", "togetherai", "bedrock"]:
            # For non-async supported models, process each request synchronously
            results = []
            for idx, messages in batch_messages:
                try:
                    response = self.generate(messages, **gen_kwargs)
                    results.append((idx, response, None))
                except Exception as e:
                    results.append((idx, None, e))
            return results

        async def _single_request(idx, messages):
            try:
                response = await self.async_generate(messages, **gen_kwargs)
                return (idx, response, None)
            except Exception as e:
                logger.warning(f"Error processing request with index {idx}: {e}")
                return (idx, None, e)

        tasks = [_single_request(idx, messages) for idx, messages in batch_messages]
        return await asyncio.gather(*tasks)

    def batch_generate_with_indices_sync(
        self,
        batch_messages: List[Tuple[Any, List[Dict[str, str]]]],
        batch_size: int = 10,
        **gen_kwargs
    ) -> List[Tuple[Any, Optional[str], Optional[Exception]]]:
        """
        Synchronous batch processing method that preserves indices/metadata.

        Args:
            batch_messages: List of tuples (index/metadata, messages)
            batch_size: Number of requests to process in each batch
            **gen_kwargs: Generation kwargs passed to the model

        Returns:
            List of tuples (index/metadata, response_or_None, exception_or_None)
        """
        results = []
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i+batch_size]
            batch_results = asyncio.run(self.batch_generate_with_indices(batch, **gen_kwargs))
            results.extend(batch_results)
        return results


if __name__ == "__main__":
    prompt = """You are an expert in crafting engaging and thought-provoking moral dilemma scenarios. Given a title and a detailed description of a situation, your task is to concisely summarize the scenario in approximately 200 words, ensuring all key details that influence the moral decision remain intact. Then, reframe the scenario into a clear binary-choice question with two distinct options, labeled A and B, each with a brief justification.

Example input format:
```
"title": "[Title of the scenario]",
"text": "[Detailed description of the situation that presents a moral dilemma]"
```

Example output format:
```
Moral dilemma question:
[Summarized scenario presenting the moral dilemma in approximately 200 words]

What do you do?
A. [Description of one choice and its justification]
B. [Description of the other choice and its justification]
```

Input:
```
"title": {title},
"text": {text}
```

Output:

"""

    # test openai model
    # model = ChatModel(model="gpt-4o-mini", provider="openai")
    model = ChatModel(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", provider="togetherai")
    response = model.generate(
        [
            {"role": "assistant", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt.format(
                    title="AITA for turning my ex in for a warrant?",
                    text="Moved in with a girlfriend two months ago who dumped me a little over a week ago. I've moved back in with my parents and let her keep the place because she agreed to pay the rent and pay me some money she owed me by the first of February. The apartment complex contacted me today and let me know the rent hasn't been paid, she also hasn't paid me back and has been ignoring me. I want to turn her in while she's at work. ",
                ),
            },
        ]
    )
    print(response)

    # Test batch processing
    import asyncio

    async def test_batch():
        # Create batch messages
        batch_messages = [
            [
                {"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about Python programming language."}
            ],
            [
                {"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is artificial intelligence?"}
            ]
        ]

        results = await model.batch_generate(batch_messages)
        print("Batch results:", results)

        # Test sync version
        # sync_results = model.batch_generate_sync(batch_messages)
        # print("Sync batch results:", sync_results)

    # Run async test
    asyncio.run(test_batch())

    # test llama3 70B
    # model = ChatModel(
    #     model="meta-llama/Llama-3.3-70B-Instruct", provider="huggingface", cache_dir=""
    # )
