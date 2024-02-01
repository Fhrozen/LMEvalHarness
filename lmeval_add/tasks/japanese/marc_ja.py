"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is 
built to measure the general NLU ability in Japanese.
JGLUE has been constructed from scratch without translation.

Homepage: https://github.com/yahoojapan/JGLUE
"""

import os
import numpy as np
from lmeval_add.api.task import MultipleChoiceTask
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean, matthews_corrcoef

from lmeval_add.utils.metrics import balanced_mean, macro_f1

_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, 
    it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. 
    While the English NLU benchmark, GLUE, has been the forerunner, 
    benchmarks are now being released for languages other than English, 
    such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. 
    We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure 
    the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""

@register_task("marc_ja_1.1-0.2")
class MARCJaWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?]
    (https://fintan.jp/page/9126/)
    """

    VERSION = 1.1
    PROMPT_VERSION = 0.2
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "MARC-ja"
    description = """製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。
    出力は小文字化してください。 \n\n"""
    CHOICES = ["positive", "negative"]
    SEP = "\n"
    num_fewshot = 1

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "query": doc["sentence"],
            "choices": self.CHOICES,
            "gold": int(doc["label"]),
        }

    def doc_to_text(self, doc):
        """
        製品レビュー:{query}
        センチメント:
        """
        return f"製品レビュー:{doc['query']}\n" "センチメント:"

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    # def construct_requests(self, doc, ctx):
    #     lls = [
    #         rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
    #     ]

    #     # this is only used for error analysis
    #     if os.environ.get("DEBUG_MULTIPLECHOICE"):
    #         lls.append(rf.greedy_until(ctx, [self.SEP]))

    #     return lls
    def construct_requests(self, doc, ctx, **kwargs):
        # rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        lls = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, "{}".format(choice)),
                idx=idx,
                **kwargs,      
            ) for idx, choice in enumerate(doc["choices"])
        ]
        # this is only used for error analysis
        if os.environ.get("DEBUG_MULTIPLECHOICE"):
            lls.append(Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, dict(until=[self.SEP])),
                idx=0,
                **kwargs
            ))
            # lls.append(rf.greedy_until(ctx, [self.SEP]))
        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        # This isn't very clean, but it may be the best we can do since lm ops
        # are submitted as an iterator for batching
        response = None
        if isinstance(results[-1], str):
            response = results.pop()

        pred = np.argmax(results)
        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "balanced_acc": (acc, gold),
            "mcc": (gold, pred),
            "macro_f1": (gold, pred),
            # "details": {
            #     "question": self.doc_to_text(doc),
            #     "response": response,
            #     "scores": results,
            # },
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "balanced_acc": balanced_mean,
            "mcc": matthews_corrcoef,
            "macro_f1": macro_f1,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "balanced_acc": True,
            "mcc": True,
            "macro_f1": True,
        }


@register_task("marc_ja_1.1-0.3")
class MARCJaWithJAAlpacaPrompt(MARCJaWithFintanPrompt):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data.
    ```
    {
        'instruction': '以下のテキストを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。',
        'input': '製品が遅すぎて使い勝手が悪かったので、あまり好きではありませんでした。',
        'output': 'ネガティブ。'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/
        c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """

    PROMPT_VERSION = 0.3
    description = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。
    要求を適切に満たす応答を書きなさい。\n\n"""
    INSTRUCTION = "以下の製品レビューを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。"
    CHOICES = ["ポジティブ", "ネガティブ"]

    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {instruction}

        ### 入力:
        {input}

        ### 応答:
        {response}
        """
        input_text = doc["query"]
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


@register_task("marc_ja_1.1-0.4")
class MARCJaWithRinnaInstructionSFT(MARCJaWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    description = (
        """ユーザー: 与えられた製品レビューを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。
        <NL>システム: 分かりました。<NL>"""
    )
    CHOICES = ["ポジティブ", "ネガティブ"]
    SEP = "<NL>"
    fewshot_delimiter = "<NL>"

    def doc_to_text(self, doc):
        input_text = doc["query"]
        return f"ユーザー: {input_text}{self.SEP}システム: "


@register_task("marc_ja_1.1-0.5")
class MARCJaWithRinnaBilingualInstructionSFT(MARCJaWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    description = (
        """ユーザー: 与えられた製品レビューを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。
        システム: 分かりました。\n"""
    )
    SEP = "\n"
    fewshot_delimiter = "\n"


@register_task("marc_ja_1.1-0.6")
class MARCJaWithLlama2(MARCJaWithJAAlpacaPrompt):
    """
    This prompt version follows the Llama2-chat's prompt format:
    ```
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    ```
    reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    PROMPT_VERSION = 0.6
    # DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. 
    # Always answer as helpfully as possible, while being safe.  
    # Your answers should not include any harmful, unethical, racist, sexist, 
    # toxic, dangerous, or illegal content. Please ensure that your 
    # responses are socially unbiased and positive in nature.
    # If a question does not make any sense, or is not factually coherent, 
    # explain why instead of answering something not correct. 
    # If you don't know the answer to a question, please don't share false information."""

    DEFAULT_SYSTEM_PROMPT = "あなたは役立つアシスタントです。"
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    description = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    fewshot_delimiter = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`, which is based on prompt version 0.3
        ```
        以下の製品レビューを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。

        {query} [/INST]
        ```
        """
        input_text = doc["query"]
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "
