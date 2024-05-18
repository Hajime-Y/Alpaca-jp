"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils_di as utils

import fire

from transformers import AutoTokenizer

# tokenizer の dead lock warning を回避
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# RougeScorerの日本語化対応
hf_tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-MS-7b-v0.1")

def encode_prompt(prompt_instructions, base_prompt_file="./prompt_en_for_jp.txt"):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(base_prompt_file).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response.message.content
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response.finish_reason == "length":
            print("Finish reasonがlengthです")
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        print(splitted_data)
        # splitted_data[3]に"Instruction"が来ている場合、[0], [1]を削除し、indexを前にずらす
        # ['', 'Instruction', '', 'Instruction', ' この英文を日本語に翻訳...] みたいなことがあるため
        if len(splitted_data) > 3:
            if splitted_data[3] == "Instruction":
                splitted_data = splitted_data[2:]  # [0], [1]を削除し、indexを前にずらす
        if len(splitted_data) != 7:
            print("Fileter: splitted_data length is not 7.")
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst) <= 6 or len(inst) > 300:
            print("Filter: too short or too long instructions. instruction lengths: ", len(inst.split()))
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            print("Filter: based on keywords that are not suitable for language models.")
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            print("Filter: Write a program isというテキストからInstructionが始まっています")
            continue
        # filter those starting with punctuation
        # 日本語の場合、""のような特殊文字が最初に来る場合もあるはず
        if inst[0] in string.punctuation.replace("'", "").replace('"', ""):
            print("Filter: 句読点で始まっています")
            continue
        # filter those starting with non-english character
        # if not inst[0].isascii():
        #     continue
        # instがすべて英語のASCII文字である場合
        if all(char.isascii() for char in inst):
            print("Filter: instはすべて英語のASCII文字です")
            continue
        # instに英語と日本語以外の文字が含まれている場合
        if any(not (char.isascii() or utils.is_japanese(char)) for char in inst):
            print("Filter: instに英語と日本語以外の文字が含まれています")
            continue
        # inputが記号だけの場合（input==""の場合は除く）
        if input and all(char in string.punctuation for char in input):
            print("Filter: inputが記号だけです")
            continue
        new_inst = {"index": idx, "instruction": inst, "input": input, "output": output}
        print(f"new_inst: {new_inst}")
        instructions.append(new_inst)
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    # output_dir="./",
    output_path="./regen_di.json",
    seed_tasks_path="./seed_tasks/seed_tasks_jp_cleaned.jsonl",
    num_instructions_to_generate=100,
    model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",  # mistralai/Mixtral-8x22B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, cognitivecomputations/dolphin-2.6-mixtral-8x7b
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    base_prompt_file="./prompt_en_for_jp.txt",
    max_rouge_scores=0.7,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(output_path):
        machine_instruction_data = utils.jload(output_path)
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=hf_tokenizer, use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, base_prompt_file=base_prompt_file)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=2048,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n10", "10.", "10."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > max_rouge_scores:
                print(f"\nFilter: 最大rouge_scoresが{max_rouge_scores}を超えています。")
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, output_path)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
