import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union, Dict

from mistralai.client import MistralClient
from mistralai.models.chat_completion import (
    ChatMessage,
    ResponseFormat,
    ToolChoice,
)
import tqdm
import copy


with open('config.json') as f:
    config = json.load(f)

api_key = config['MISTRAL_API_KEY']
if api_key == "" or api_key is None:
    logging.warning(f"api_key isn't set.")


@dataclasses.dataclass
class MistralDecodingArguments:
    temperature: float = 0.2
    max_tokens: int = 1800
    top_p: float = 1.0
    random_seed: int = 42
    safe_mode: bool = False
    safe_prompt: bool = False
    tool_choice: Optional[Union[str, ToolChoice]] = None
    response_format: Optional[Union[Dict[str, str], ResponseFormat]] = None


def mistralai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: MistralDecodingArguments,
    model_name="open-mixtral-8x7b",  # model_name: open-mistral-7b, open-mixtral-8x7b, open-mixtral-8x22b
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[str, Sequence[str], Sequence[Sequence[str]]]:
    """Decode with MistralAI API.

    Args:
        prompts: A string or a list of strings to complete.
        decoding_args: Decoding arguments.
        max_instances: Maximum number of prompts to decode.
        return_text: If True, return text instead of full completion object.
        decoding_kwargs: Additional decoding arguments.

    Returns:
        A completion or a list of completions.
    """
    client = MistralClient(api_key=api_key)
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
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completions_batch = []
                # TODO: 並列実行
                for prompt in tqdm.tqdm(prompt_batch):
                    print("＝＝＝＝＝＝＝＝＝＝＝＝＝")
                    # print("prompt: ", prompt)
                    chat_response = client.chat(
                        messages=[
                            ChatMessage(role="system", content="あなたは誠実で優秀な日本人のアシスタントです。"),
                            ChatMessage(role="user", content=prompt),
                        ],
                        **shared_kwargs
                    )
                    completions_batch.append(chat_response.choices[0])
                    print("response: ", chat_response.choices[0].message.content)
                completions.extend(completions_batch)
                break
            except Exception as e:
                logging.warning(f"Error: {e}.")
                print(f"Error: {e}.")
                time.sleep(sleep_time)
                # if "Please reduce your prompt" in str(e):
                #     batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                #     logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                # else:
                #     logging.warning("Hit request rate limit; retrying...")
                #     time.sleep(sleep_time)

    if return_text:
        completions = [completion.text for completion in completions]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        completions = completions[0] if completions else None
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
        '\u3000' <= char <= '\u303F'      # Japanese Punctuation
    )