
CHOICES = {1: "はい", 0: "いいえ"}

INSTRUCTION = "与えられた文が文法的であるかを回答してください。\n\n出力は以下から選択してください：\n" 
INSTRUCTION += "\n".join(
    list(CHOICES.values())
)


def doc2text_03(doc):
    input_text = doc["sentence"]
    return f"### 指示:\n{INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


def doc2text_06(doc):
    input_text = doc["sentence"]
    return f"{INSTRUCTION}\n\n{input_text} [/INST] "
