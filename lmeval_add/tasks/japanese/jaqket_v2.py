"""
JAQKET: JApanese Questions on Knowledge of EnTitie
https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf


Homepage: https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
"""
import os
import inspect
import datasets
import evaluate
from functools import partial
from math import exp

from lmeval_add.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task

from lmeval_add.jasquad import jasquad

_CITATION = """
@InProceedings{Kurihara_nlp2020,
  author =  "鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也",
  title =   "JAQKET: クイズを題材にした日本語 QA データセットの構築",
  booktitle =   "言語処理学会第26回年次大会",
  year =    "2020",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf"
  note= "in Japanese"
"""

TOP_K_LIMIT = 5
DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()


@register_task("jaqket_v2_0.2-0.1")
class JAQKETV2(Task):
    """
    prompt template is taken from 
    [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価]
    (https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """

    VERSION = 0.2
    PROMPT_VERSION = 0.1
    DATASET_PATH = "kumapo/JAQKET"
    DATASET_NAME = "v2.0"
    LOAD_TOKENIZER = True
    SEP = "\n"
    REMOVE_IDS = []
    max_length = None
    description = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
    num_fewshot = 1
    dataset_kwargs={
        "num_contexts" : TOP_K_LIMIT,
    }
    jasqaud_metric = evaluate.load(jasquad.__file__)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        dataset = self.dataset["validation"]
        if len(self.REMOVE_IDS) > 0:
            dataset = [item for item in dataset if item["id"] not in self.REMOVE_IDS]
        return dataset

    def doc_to_qa_prompt(self, doc):
        return "[質問]:" + doc["question"] + self.SEP + "[答え]:"

    def doc_to_text(self, doc):
        answer_candidate = self.SEP.join(
            [
                ("[題名]:" + title + self.SEP + "[問題]:" + context)
                for title, context in zip(doc["ctxs"]["title"], doc["ctxs"]["text"])
            ]
        )
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = (
            "[題名]:"
            + answering_contexts["title"][0]
            + self.SEP
            + "[問題]:"
            + answering_contexts["text"][0]
        )
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        answer = answer_list[0]
        return answer

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        max_num_tokens = max(
            [len(self._tokenize(answer)) for answer in doc["answers"]["text"]]
        )
        max_length = self.max_length - max_num_tokens

        # If the prompt is too long with fewshot examples, reduce the number of
        # examples until it fits.
        while num_fewshot >= 0:
            ctx = super().fewshot_context(doc, num_fewshot, **kwargs)
            if len(self._tokenize(ctx)) <= max_length:
                doc["context"] = ctx
                return ctx
            num_fewshot -= 1

        # if we got here then even 0 fewshot is too long
        return ValueError(
            f"0-shot prompt is too long for max length {max_length}:\n{ctx}"
        )

    def _tokenize(self, text, **kwargs):
        encode_fn = self.tokenizer.encode
        encode_params = dict()
        if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
            encode_params["add_special_tokens"] = False

        return encode_fn(text, **encode_params, **kwargs)

    def construct_requests(self, doc, ctx, **kwargs):
        args_until = dict(until=[self.SEP])
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            # continuation = rf.greedy_until(ctx, [self.SEP])
            pass
        else:
            # continuation = rf.greedy_until(ctx, [self.SEP], max_num_tokens)
            encode_fn = self.tokenizer.encode
            encode_params = dict()
            if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
                encode_params["add_special_tokens"] = False
   
            max_num_tokens = max(
                [
                    len(encode_fn(answer, **encode_params))
                    for answer in doc["answers"]["text"]
                ]
            )
            args_until.update(max_gen_toks=max_num_tokens)

        continuation = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, args_until),
            idx=0,
            **kwargs
        )
        return continuation

    def process_results(self, doc, results):
        assert (
            len(results) == 1
        ), f"results should be a list with 1 str element, but is {results}"
        continuation = results[0]
        predictions = {
            "id": doc["qid"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["qid"],
            "answers": doc["answers"],
        }
        out = {
            "exact_match": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
        }

        # add details. Because the metric computation isn't simple (probably?)
        # always include it.
        # out["details"] = {
        #     "question": doc["question"],
        #     "response": continuation,
        #     "gold": doc["answers"],
        # }

        return out

    def aggregation(self):
        return {
            "exact_match": partial(
                self._squad_agg, "exact_match"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                self._squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        return {
            "exact_match": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }

    def _squad_metric(self, predictions, references):
        return self.jasqaud_metric.compute(
            predictions=predictions, references=references
        )

    def _squad_agg(self, key, item):
        predictions, references = zip(*item)
        return self._squad_metric(predictions=predictions, references=references).get(key, 0)


@register_task("jaqket_v2_0.2-0.2")
class JAQKETV2WithFintanPrompt(JAQKETV2):
    """
    prompt template is taken from [ChatGPT vs BERT: 
    どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    PROMPT_VERSION = 0.2
    description = "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。\n\n"
    SEP = "\n"

    def doc_to_qa_prompt(self, doc):
        return "質問:" + doc["question"] + self.SEP + "回答:"

    def doc_to_text(self, doc):
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文章:" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文章:" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return answer_candidate + self.SEP + qa_prompt


@register_task("jaqket_v2_0.2-0.3")
class JAQKETV2WithJAAlpacaPrompt(JAQKETV2):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data.
    ```
    {
        'instruction': '与えられた文脈に最も適した文を選択してください。',
        'input': '文脈：あなたは親友と現在の仕事の状況について話しています。
        A）私にはあまり選択肢がありません。
        B）他に選択肢がありません。
        C）私には本当に決断する必要がありません。',
        'output': 'A) 私には多くの選択肢がありません。'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/
        c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """

    PROMPT_VERSION = 0.3
    description = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。
    要求を適切に満たす応答を書きなさい。"""
    INSTRUCTION = "与えられた文脈から、質問に対する答えを抜き出してください。"

    def doc_to_qa_prompt(self, doc):
        return "質問：" + doc["question"]

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
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{answer_candidate}\n{qa_prompt}\n\n### 応答:\n"

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{answer_candidate}\n{qa_prompt}\n\n### 応答:\n"


@register_task("jaqket_v2_0.2-0.4")
class JAQKETV2WithRinnaInstructionSFT(JAQKETV2):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    PROMPT_VERSION = 0.4
    SEP = "<NL>"
    END_OF_DESCRIPTION = "システム: 分かりました。<NL>"
    START_OF_FEWSHOT = "ユーザー: 文脈："
    fewshot_delimiter = "<NL>"

    def doc_to_qa_prompt(self, doc):
        return "質問：" + doc["question"]

    def doc_to_text(self, doc):
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}{qa_prompt}{self.SEP}システム: "

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"ユーザー: {answer_candidate}{self.SEP}{qa_prompt}{self.SEP}システム: "


@register_task("jaqket_v2_0.2-0.5")
class JAQKETV2WithRinnaBilingualInstructionSFT(JAQKETV2WithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    description = "ユーザー: 与えられた文脈から、質問に対する答えを抜き出してください。\nシステム: 分かりました。\n"
    SEP = "\n"
    fewshot_delimiter = "\n"


@register_task("jaqket_v2_0.2-0.6")
class JAQKETV2WithLlama2(JAQKETV2WithJAAlpacaPrompt):
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
    # This is the English prompt.
    # DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. 
    # Always answer as helpfully as possible, while being safe.
    # Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, 
    # or illegal content. Please ensure that your responses are socially 
    # unbiased and positive in nature.
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
        与えられた文脈から、質問に対する答えを抜き出してください。

        文脈：{context}
        質問：{question} [/INST]
        ```
        """
        context = self.SEP.join([text for text in doc["ctxs"]["text"]])
        answer_candidate = "文脈：" + context
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"{self.INSTRUCTION}\n\n{answer_candidate}\n{qa_prompt} [/INST] "

    def doc_to_answering_text(self, doc):
        has_answer = doc["ctxs"]["has_answer"]
        answering_index = has_answer.index(True)
        answering_contexts = {
            k: v[answering_index : answering_index + 1] for k, v in doc["ctxs"].items()
        }
        answer_candidate = "文脈：" + answering_contexts["text"][0]
        qa_prompt = self.doc_to_qa_prompt(doc)
        return f"{self.INSTRUCTION}\n\n{answer_candidate}\n{qa_prompt} [/INST] "
