import re


def clean_json_output(text: str) -> str:
    """
    清理 Gemini 2.5 模型輸出的 JSON 字串，移除前綴和後綴的標記。
    """
    # 移除開頭和結尾的 ```json 及可能的換行符號
    cleaned_text = re.sub(r"^```json\n?", "", text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"```$", "", cleaned_text)

    # 移除開頭和結尾的 ``` 符號，不區分大小寫
    cleaned_text = re.sub(r"^```\n?", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"```$", "", cleaned_text, flags=re.IGNORECASE)

    # 移除字串開頭和結尾的空白字元
    cleaned_text = cleaned_text.strip()

    return cleaned_text