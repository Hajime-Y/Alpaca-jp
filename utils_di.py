import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

from openai import OpenAI, OpenAIError
import tqdm
import copy

import concurrent.futures


with open('config.json') as f:
    config = json.load(f)

api_key = config['DEEPINFRA_API_KEY']
base_url = "https://api.deepinfra.com/v1/openai"
if api_key == "" or api_key is None:
    logging.warning(f"api_key isn't set.")


# refers: https://deepinfra.com/mistralai/Mixtral-8x22B-Instruct-v0.1/api?example=openai-python
@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    # presence_penalty: float = 0.0
    # frequency_penalty: float = 0.0
    # suffix: Optional[str] = None
    # logprobs: Optional[int] = None
    # echo: bool = False


# APIの呼び出しを行う関数
def fetch_completion(client, prompt, shared_kwargs):
    try:
        chat_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a useful and honest AI assistant."},
                {"role": "user", "content": prompt}
            ],
            **shared_kwargs
        )
        choice = chat_response.choices[0]
        return choice
    except OpenAIError as e:
        print(f"OpenAIError: {e}.")
        logging.warning(f"OpenAIError: {e}.")
        return e  # 例外を返す


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[str, Sequence[str], Sequence[Sequence[str]]]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            shared_kwargs = dict(
                model=model_name,
                **batch_decoding_args.__dict__,
                **decoding_kwargs,
            )
            completions_batch = []
            
            # APIコールの並列実行
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # プロンプトごとに実行    
                future_to_prompt = {executor.submit(fetch_completion, client, prompt, shared_kwargs): prompt for prompt in prompt_batch}
                errors = []
                for future in concurrent.futures.as_completed(future_to_prompt):
                    result = future.result()
                    if isinstance(result, OpenAIError):
                        errors.append(result)
                    elif result is not None:
                        completions_batch.append(result)
                
            if completions_batch:
                completions.extend(completions_batch)
                break
            elif errors:
                # エラーハンドリング
                for error in errors:
                    if "Please reduce your prompt" in str(error):
                        batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                        print(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                        logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                        break  # 再試行のためにループを抜ける
                    elif "rate limit" in str(error):
                        print("Hit request rate limit; retrying...")
                        logging.warning("Hit request rate limit; retrying...")
                        time.sleep(sleep_time)  # リクエストレートリミットのために一時停止
                        break  # 再試行のためにループを抜ける
                else:
                    # その他のエラーは単にログに記録
                    print("Unhandled error occurred, logging and continuing...")
                    logging.error(f"Unhandled error: {error}")

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str, ensure_ascii=False):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def is_japanese(char):
    """
    指定された文字が日本語の文字であるかどうかを判定します。
    refers: https://zenn.dev/shundeveloper/articles/a4be0379508e2d

    Args:
        char (str): 判定する単一の文字。

    Returns:
        bool: 文字が日本語の場合はTrue、それ以外の場合はFalse。
    """
    return (
        '\u3040' <= char <= '\u309F' or  # Hiragana
        '\u30A0' <= char <= '\u30FF' or  # Katakana
        '\uFF65' <= char <= '\uFF9F' or  # Half-width Katakana
        '\u31F0' <= char <= '\u31FF' or  # Katakana Phonetic Extensions
        '\u4E00' <= char <= '\u9FFF' or  # CJK Unified Ideographs
        '\u3400' <= char <= '\u4DBF' or  # CJK Extension A
        '\u20000' <= char <= '\u2A6DF' or  # CJK Extension B
        '\u2A700' <= char <= '\u2B73F' or  # CJK Extension C
        '\u2B820' <= char <= '\u2CEAF' or  # CJK Extension E
        '\u2CEB0' <= char <= '\u2EBEF' or  # CJK Extension F
        '\u3000' <= char <= '\u303F' or   # Japanese Punctuation
        '\uFF01' <= char <= '\uFF5E'      # Full-width ASCII variants including full-width question mark
    )