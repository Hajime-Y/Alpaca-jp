# Alpaca-jp

[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)データセットの日本語版をMixtral-8x22B, Mixtral-8x7Bを用いて作成するためのコードです。

## Usage
 - MistralAIのAPIを用いて作成: [10_generate_instruction.ipynb](https://github.com/Hajime-Y/Alpaca-jp/blob/main/10_generate_instruction.ipynb)
 - deepinfra APIを用いて作成: [11_generate_instruction_di.ipynb](https://github.com/Hajime-Y/Alpaca-jp/blob/main/11_generate_instruction_di.ipynb)
 - 合成データのクレンジング: [12_clean_instructions.ipynb](https://github.com/Hajime-Y/Alpaca-jp/blob/main/12_clean_instructions.ipynb)

## Seed tasks

デフォルトでは[seed_tasks_jp_cleaned.jsonl](https://github.com/Hajime-Y/Alpaca-jp/blob/main/seed_tasks/seed_tasks_jp_cleaned.jsonl)を使用しています。  
これは、以下の手順で作成しました。
 - Stanford Alpacaの[seed_tasks.jsonl](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl)をDeepLを用いて翻訳
 - 手作業で修正  

## Prompt

データ合成のためのプロンプトはデフォルトでは[prompt_en_for_jp.txt](https://github.com/Hajime-Y/Alpaca-jp/blob/main/prompt_en_for_jp.txt)を使用します。  
元々、日本語のプロンプトである[prompt_jp.txt](https://github.com/Hajime-Y/Alpaca-jp/blob/main/prompt_jp.txt)の利用を検討しましたが、Mixtralの場合英語のバイアスが強く英語で指示した方が日本語で返してくれることが多いと感じたためprompt_en_for_jp.txtを使用しました。

## Data Cleaning

データクリーニングには以下のプロンプトを使用しています。  
変更する場合は、[data_cleaning.py](https://github.com/Hajime-Y/Alpaca-jp/blob/main/data_cleaning.py)を修正ください。
```Python
text = f"""Assess whether the following combination of instruction, input, and output is appropriate. 
1. Ensure the instruction is clear and only in Japanese.
2. Verify that the input data matches the language and context of the instruction.
3. Check the output data for:
    - Language consistency with the instruction and input.
    - Accuracy and relevance to the input.
    - Clarity without repetition or errors.
Return True or False.
\nInstruction: {instruction}\nInput: {input_data}\nOutput: {output_data}"""
```