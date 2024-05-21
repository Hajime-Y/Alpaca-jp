from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm.auto import tqdm
import json
import time
import os
import re
import sys
import subprocess

from data_cleaning import load_json_from_file, create_output_path

with open('config.json') as f:
    config = json.load(f)

openai = OpenAI(
    api_key=config['DEEPINFRA_API_KEY'],
    base_url="https://api.deepinfra.com/v1/openai",
)

# ========================================
# python実行用関数
# ========================================
def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)

def culc_code_result(output, index=""):
    code_output = None

    try:
        # outputから「<llm-code>」と「</llm-code>」で囲まれた全てのコードを抜き出す。
        codes = re.findall(r'<llm-code>(.*?)</llm-code>', output, re.DOTALL)
        # 最後のコード以外からprint文を削除
        cleaned_codes = [re.sub(r'\nprint\(.*?\)$', '', code, flags=re.MULTILINE) if i != len(codes)-1 else code for i, code in enumerate(codes)]
        # 全てのコードを改行で結合
        code = '\n'.join(cleaned_codes)

        # code.pyへの書き込み
        filename = f"math_code_check_{index}.py"
        with open(filename, 'w') as fout:
            fout.write(code)

        # code.py を最大7秒間実行。出力はUTF-8でデコード。
        # batcmd = 'timeout 7 ' + sys.executable + ' code.py'
        batcmd = [sys.executable, filename]  # local
        try:
            # shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
            shell_output = subprocess.check_output(batcmd, timeout=7, stderr=subprocess.STDOUT).decode('utf8')  # local
            code_output = round(float(eval(shell_output)), 2)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred. Error code: {e.returncode}, Output: {e.output.decode('utf-8')}")
            output_message = e.output.decode('utf-8')
            if "ImportError" in output_message or "ModuleNotFoundError" in output_message:
                raise ImportError("An import error occurred in the subprocess.")
        except subprocess.TimeoutExpired:
            print("The process has timed out.")
        except Exception as e:
            print(e)

        print(' - CODE RESULTS', code_output)
    except Exception as e:
        print(e)
        print('ERROR PARSING')
    finally:
        # filenameを削除
        if os.path.exists(filename):
            os.remove(filename)

    return code_output


def get_text_result(output):
    text_output = None

    try:
        # 結果から\\boxed{...} 形式を取得
        text_outputs = re.findall(r'\\boxed\{(.*)\}', output)
        text_output = round(float(eval(text_outputs[-1])), 2)
        print(' - BOXED', text_output)
    except Exception as e:
        print(e)
        print('ERROR PARSING')

    return text_output


def compare_process_output(code_output, text_output):
    # code_outputとtext_outputが共にNoneではなく、同等であるかを出力
    return code_output is not None and text_output is not None and code_output == text_output


# ========================================
# cleanチェック付与用関数
# ========================================
def create_math_prompt(instruction, input_data, output_data):
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
        1. The only natural language for instructions and output is Japanese.
        2. The task must be math task.
        3. Verify that the input data matches the language and context of the instruction.
        4. Check the output data for:
            - Language consistency with the instruction and input.
            - Accuracy and relevance to the input.
            - Clarity without repetition or errors.
        \nInstruction: {instruction}\nOutput: {output_data}
        \nYour Judgement (Just answer: True or False. No need to explain the reason.):"""
    else:
        text = f"""Assess whether the following combination of instruction, input, and output is appropriate. 
        1. The only natural language for instructions, input, and output is Japanese.
        2. The task must be math task.
        3. Verify that the input data matches the language and context of the instruction.
        4. Check the output data for:
            - Language consistency with the instruction and input.
            - Accuracy and relevance to the input.
            - Clarity without repetition or errors.
        \nInstruction: {instruction}\nInput: {input_data}\nOutput: {output_data}
        \nYour Judgement (Just answer: True or False. No need to explain the reason.):"""
    return text

def check_tags(text):
    """
    与えられた文字列内に<llm-code></llm-code>タグの直後に<llm-code-output></llm-code-output>タグが少なくとも1つ存在するかを確認する関数。
    
    Args:
    text (str): 検査する文字列。
    
    Returns:
    bool: タグの組み合わせが正しく1つ以上存在する場合はTrue、そうでない場合はFalse。
    """
    pattern = r"<llm-code>.*?</llm-code>\s*<llm-code-output>\s*\n\s*[+-]?(\d+\.\d*|\.\d+|\d+)\s*</llm-code-output>"
    matches = re.findall(pattern, text, re.DOTALL)
    return len(matches) > 0

def check_boxed_decimal(text):
    """
    与えられたテキスト内で、LaTeX形式の\\boxedコマンドを使用して囲まれた小数を検出する関数です。
    この関数は、\\boxed{...} の中に小数が含まれているかどうかをチェックします。

    Args:
        text (str): 検査するテキスト。

    Returns:
        bool: テキスト内に適切な形式の小数が含まれていればTrue、そうでなければFalse。
    """
    # 正規表現パターン: \\boxed{文字列}
    pattern = r"\\boxed\{\s*[+-]?(\d+\.\d*|\.\d+|\d+)\s*\}"
    # search() を使用してテキスト全体でパターンを検索
    match = re.search(pattern, text)
    # マッチがあれば True、なければ False を返す
    return match is not None

def check_math_prompts(data, check_model="mistralai/Mixtral-8x22B-Instruct-v0.1"):
    """
    データに対してチェックを行い、結果をbool値で返す。

    Args:
    data (dict): チェックするデータ(instruction, input, outputを持つ)

    Returns:
    bool: チェックの結果
    """
    # タグの確認
    if not check_tags(data["output"]):
        return "False"
    # コードの計算結果とテキストの解答が同じかどうか確認
    if not compare_process_output(data["code_result"], data["text_result"]):
        return "False"
    # 最大5回確認
    for _ in range(5):
        try:
            # LLMによるチェック
            prompt = create_math_prompt(data["instruction"], data["input"], data["output"])
            chat_response = openai.chat.completions.create(
                model=check_model,
                messages=[
                    {"role": "system", "content": "You are a useful and honest AI assistant. You excel in math and python coding abilities."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3,
                temperature=0.9,
            )
            check_result = chat_response.choices[0].message.content
            print(f"check_result: {check_result}")
            
            if "True" in check_result and "False" not in check_result:
                return "True"
            elif "True" not in check_result and "False" in check_result:
                return "False"
            else:
                continue
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)  # リクエストレートリミットのために一時停止
    # 最大数確認しても結果が不明な場合、Falseとする
    return "False"

def add_clean_to_dict_math(input_path='./regen_math.json', previous_version_path="", check_all=False, check_model="mistralai/Mixtral-8x22B-Instruct-v0.1"):
    """
    JSONファイルを読み込み、データに'clean'キーを追加して、新しいパスに保存する。
    過去バージョンのファイルが指定されている場合、そのファイルから'clean'の値を取得して使用する。

    Args:
    input_path (str): 入力ファイルのパス
    previous_version_path (str): 過去バージョンのファイルパス
    """
    data_list = load_json_from_file(input_path)

    # codeの計算結果とtextから読み取れる計算を取得する
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for index, data in enumerate(data_list):
            output = data["output"]
            # すでに結果が存在しない場合のみ計算を行う
            if "code_result" not in data or "text_result" not in data:  # すでに結果が存在しない場合のみ計算を行う
                future_code = executor.submit(culc_code_result, output, str(index))
                future_text = executor.submit(get_text_result, output)
                futures.append((future_code, future_text, data))

        for future_code, future_text, data in futures:
            data["code_result"] = future_code.result()
            data["text_result"] = future_text.result()

    # 既存cleanファイルの読み込み
    if previous_version_path:
        previous_data_list = load_json_from_file(previous_version_path)
        previous_clean_dict = {data['instruction']: data.get('clean', 'False') for data in previous_data_list}

    # 過去のcleanファイルが渡された場合、そこからcleanの結果を読み込む
    # 同じレコードかどうかはそのレコードの"instruction"を確認する
    def process_data(data, check_model):
        if check_all:
            return check_math_prompts(data, check_model)
        if 'clean' in data and data['clean'] in ["True", "False"]:
            return data['clean']
        elif previous_version_path and data['instruction'] in previous_clean_dict:
            return previous_clean_dict[data['instruction']]
        else:
            return check_math_prompts(data, check_model)
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_data, data, check_model): data for data in data_list}
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