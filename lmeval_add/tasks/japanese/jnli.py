"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese.
JGLUE has been constructed from scratch without translation.

Homepage: https://github.com/yahoojapan/JGLUE
"""
import os
import numpy as np

# from lm_eval.base import BalancedMultipleChoiceTask, rf
from lm_eval.api.task import MultipleChoiceTask
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
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""

@register_task("jnli_1.3-0.2")
class JNLIWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    VERSION = 1.3
    PROMPT_VERSION = 0.2
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JNLI"
    DESCRIPTION = (
        "前提と仮説の関係を含意、矛盾、中立の中から回答してください。\n\n"
        + "制約:\n"
        + "- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合は含意と出力\n"
        + "- 前提と仮説が両立しえない場合は矛盾と出力\n"
        + "- そのいずれでもない場合は中立と出力\n\n"
    )
    CHOICES = ["含意", "矛盾", "中立"]
    SEP = "\n"

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
            "premise": doc["sentence1"],
            "hypothesis": doc["sentence2"],
            "choices": self.CHOICES,
            "gold": int(doc["label"]),
        }

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
            "details": {
                "question": self.doc_to_text(doc),
                "response": response,
                "scores": results,
            },
        }

    def doc_to_text(self, doc):
        """
        前提:{premise}
        仮説:{hypothesis}
        関係:
        """
        return f"前提:{doc['premise']}\n" f"仮説:{doc['hypothesis']}\n" "関係:"

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "balanced_acc": True,
            "mcc": True,
            "macro_f1": True,
        }

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

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "balanced_acc": balanced_mean,
            "mcc": matthews_corrcoef,
            "macro_f1": macro_f1,
        }


@register_task("jnli_1.3-0.3")
class JNLIWithJAAlpacaPrompt(JNLIWithFintanPrompt):
    """
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """

    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = f"与えられた前提と仮説の関係を回答してください。\n\n出力は以下から選択してください：\n" + "\n".join(
        JNLIWithFintanPrompt.CHOICES
    )

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
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


@register_task("jnli_1.3-0.4")
class JNLIWithRinnaInstructionSFT(JNLIWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    DESCRIPTION = (
        "ユーザー: "
        + f"与えられた前提と仮説の関係を回答してください。出力は以下から選択してください：<NL>"
        + "<NL>".join(JNLIWithFintanPrompt.CHOICES)
        + "<NL>システム: 分かりました。<NL>"
    )
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"前提：{doc['premise']}{self.SEP}仮説：{doc['hypothesis']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


@register_task("jnli_1.3-0.5")
class JNLIWithRinnaBilingualInstructionSFT(JNLIWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    DESCRIPTION = (
        "ユーザー: "
        + f"与えられた前提と仮説の関係を回答してください。出力は以下から選択してください：\n"
        + "\n".join(JNLIWithFintanPrompt.CHOICES)
        + "\nシステム: 分かりました。\n"
    )
    SEP = "\n"
    FEWSHOT_SEP = "\n"


@register_task("jnli_1.3-0.6")
class JNLIWithLlama2(JNLIWithJAAlpacaPrompt):
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
    # DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    DEFAULT_SYSTEM_PROMPT = "あなたは役立つアシスタントです。"
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    DESCRIPTION = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    FEWSHOT_SEP = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`, which is based on prompt version 0.3
        ```
        与えられた前提と仮説の関係を回答してください。

        出力は以下から選択してください：
        含意
        矛盾
        中立

        前提：{premise}
        仮説：{hypothesis} [/INST]
        ```
        """
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"{self.INSTRUCTION}\n\n{input_text} [/INST] "
