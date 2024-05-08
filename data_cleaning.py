from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm.auto import tqdm
import json
import time
import os

with open('config.json') as f:
    config = json.load(f)

openai = OpenAI(
    api_key=config['DEEPINFRA_API_KEY'],
    base_url="https://api.deepinfra.com/v1/openai",
)

def load_json_from_file(file_path):
    """
    指定されたファイルパスからJSONデータを読み込み、Pythonの辞書として返します。

    Args:
        file_path (str): 読み込むJSONファイルのパス

    Returns:
        dict: ファイルから読み込まれたJSONデータ
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_output_path(input_path):
    """
    入力されたファイルパスから出力ファイルパスを生成する。
    入力パスのファイル名に '_cleaned' を追加して新しいパスを作成する。

    Args:
    input_path (str): 入力ファイルのパス

    Returns:
    str: 出力ファイルのパス
    """
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_cleaned{ext}")
    return output_path

def create_prompt(instruction, input_data, output_data):
    """
    指示、入力データ、出力データを組み合わせてプロンプトを作成する。

    Args:
    instruction (str): ユーザーからの指示
    input_data (str): 入力データ
    output_data (str): 出力データ

    Returns:
    str: 生成されたプロンプト
    """
    if input_data=="":
        text = f"""Assess whether the following combination of instruction, and output is appropriate. 
        1. Ensure the instruction is clear and only in Japanese.
        2. Verify that the input data matches the language and context of the instruction.
        3. Check the output data for:
            - Language consistency with the instruction and input.
            - Accuracy and relevance to the input.
            - Clarity without repetition or errors.
        Return True or False.
        \nInstruction: {instruction}\nOutput: {output_data}"""
    else:
        text = f"""Assess whether the following combination of instruction, input, and output is appropriate. 
        1. Ensure the instruction is clear and only in Japanese.
        2. Verify that the input data matches the language and context of the instruction.
        3. Check the output data for:
            - Language consistency with the instruction and input.
            - Accuracy and relevance to the input.
            - Clarity without repetition or errors.
        Return True or False.
        \nInstruction: {instruction}\nInput: {input_data}\nOutput: {output_data}"""
    return text

def check_prompts(data):
    """
    データに対してチェックを行い、結果をbool値で返す。

    Args:
    data (dict): チェックするデータ(instruction, input, outputを持つ)

    Returns:
    bool: チェックの結果
    """
    # 最大5回確認
    for _ in range(5):
        try:
            prompt = create_prompt(data["instruction"], data["input"], data["output"])
            chat_response = openai.chat.completions.create(
                model="mistralai/Mixtral-8x22B-Instruct-v0.1",
                messages=[
                    # {"role": "system", "content": "Hello"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3,
                temperature=0.9,
            )
            check_result = chat_response.choices[0].message.content
            print(f"check_result: {check_result}")
            # TrueとFalseのどちらもある場合
            if "True" in check_result and "False" in check_result:
                continue
            # TrueとFalseのどちらもない場合
            if "True" not in check_result and "False" not in check_result:
                continue
            # resultを返す(TrueとFalseのどちらか)
            if "True" in check_result or "False" in check_result:
                return "True" if "True" in check_result else "False"
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)  # リクエストレートリミットのために一時停止
    # 最大数確認しても結果が不明な場合、Falseとする
    return "False"

def add_clean_to_dict(input_path='./regen_di.json', previous_version_path="", check_all=False):
    """
    JSONファイルを読み込み、データに'clean'キーを追加して、新しいパスに保存する。
    過去バージョンのファイルが指定されている場合、そのファイルから'clean'の値を取得して使用する。

    Args:
    input_path (str): 入力ファイルのパス
    previous_version_path (str): 過去バージョンのファイルパス
    """
    data_list = load_json_from_file(input_path)

    # 既存cleanファイルの読み込み
    if previous_version_path:
        previous_data_list = load_json_from_file(previous_version_path)
        previous_clean_dict = {data['instruction']: data.get('clean', 'False') for data in previous_data_list}

    # 過去のcleanファイルが渡された場合、そこからcleanの結果を読み込む
    # 同じレコードかどうかはそのレコードの"instruction"を確認する
    def process_data(data):
        if check_all:
            return check_prompts(data)
        if 'clean' in data and data['clean'] in ["True", "False"]:
            return data['clean']
        elif previous_version_path and data['instruction'] in previous_clean_dict:
            return previous_clean_dict[data['instruction']]
        else:
            return check_prompts(data)
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_data, data): data for data in data_list}
        progress = tqdm(total=len(data_list))
        for future in as_completed(futures):
            data = futures[future]
            clean_status = future.result()
            data['clean'] = clean_status
            progress.update(1)  # 進捗バーを更新
        progress.close()  # 進捗バーを閉じる
    
    # 結果の出力
    output_path = create_output_path(input_path)
    with open(output_path, 'w') as file:
        json.dump(data_list, file, indent=4, ensure_ascii=False)