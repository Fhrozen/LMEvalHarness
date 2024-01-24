import os
import inspect
# import datasets
import evaluate
from functools import partial

from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lmeval_add.jasquad import jasquad

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

DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()


@register_task("jsquad_1.1-0.1")
class JSQuAD(Task):
    """
    prompt template is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """

    VERSION = 1.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JSQuAD"
    DESCRIPTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
    SEP = "\n"
    REMOVE_IDS = []
    # REMOVE_IDS = ['a10743p19q0', 'a10743p19q1', 'a10743p19q2', 'a10743p19q3', 'a13221p1q0', 'a13221p1q1', 'a13221p1q2', 'a13221p1q3', 'a14985p1q0', 'a14985p1q1', 'a14985p1q2', 'a14985p1q3', 'a14985p1q4', 'a14985p93q0', 'a14985p93q1', 'a14985p93q2', 'a14985p93q3', 'a14985p93q4', 'a1540503p36q0', 'a1540503p36q1', 'a1540503p36q2', 'a1540503p36q3', 'a1540503p36q4', 'a18783p1q0', 'a18783p3q0', 'a18783p3q1', 'a18783p3q2', 'a18783p8q0', 'a18873p25q0', 'a18873p25q1', 'a18873p25q2', 'a18873p25q3', 'a18873p26q0', 'a18873p26q1', 'a18873p26q2', 'a20898p10q0', 'a20898p15q0', 'a20898p15q1', 'a20898p15q2', 'a20898p15q3', 'a2164640p22q0', 'a2164640p22q1', 'a2164640p22q2', 'a2164640p22q3', 'a2164640p22q4', 'a22392p20q0', 'a22392p20q1', 'a22392p20q2', 'a22392p20q3', 'a3011628p3q0', 'a3011628p3q1', 'a3011628p3q2', 'a3011628p3q3', 'a3189p4q0', 'a3189p4q1', 'a3189p4q2', 'a369953p0q0', 'a369953p0q1', 'a369953p0q2', 'a369953p0q3', 'a3949p1q0', 'a3949p1q1', 'a4596p0q0', 'a4596p0q1', 'a4596p0q2', 'a4596p0q3', 'a4596p1q0', 'a4596p1q1', 'a4596p1q2', 'a4596p1q3', 'a4596p1q4', 'a4596p38q0', 'a4596p38q1', 'a4596p38q2', 'a4596p38q3', 'a4596p38q4', 'a4768p13q0', 'a4768p13q1', 'a4768p13q2', 'a4768p3q0', 'a4768p3q1', 'a4768p3q2', 'a4768p3q3', 'a4768p8q0', 'a4768p8q1', 'a4768p8q2', 'a51481p0q0', 'a51481p0q1', 'a51481p0q2', 'a51481p10q0', 'a51481p10q1', 'a51481p10q2', 'a51481p10q3', 'a51481p6q0', 'a51481p6q1', 'a51481p6q2', 'a51481p6q3', 'a51481p7q0', 'a51481p7q1', 'a67892p11q0', 'a67892p11q1', 'a67892p11q2', 'a67892p11q3', 'a67892p2q0', 'a8874p6q0', 'a8874p6q1', 'a916079p3q0', 'a916079p3q1', 'a95156p4q0', 'a95156p4q1', 'a95156p4q2', 'a95156p4q3', 'a95156p6q0', 'a95156p6q1', 'a95156p6q2', 'a95156p6q3']
    """
    @mkshing's comment
    I found that JSQuAD contains errors inside contexts such as below.
    ```
    {'id': 'a4596p0q0', 'title': 'ポルトガル', 'context': 'ポルトガル [SEP] 正式名称はポルトガル語で、。通称、 。', 'question': 'ポルトガルね正式名称は何語であるか', 'answers': {'text': ['正式名称はポルトガル語', 'ポルトガル語', 'ポルトガル語'], 'answer_start': [12, 17, 17]}, 'is_impossible': False}
    ```
    So, I tried to identify all of them and found that the following processing can be okay to detect the ids
    ```python
    from datasets import load_dataset
    from transformers import T5Tokenizer
    dataset = load_dataset("shunk031/JGLUE", name="JSQuAD", split="validation")
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    remove_ids = []
    for item in dataset:
        ctx = item["context"].split("[SEP]")[-1].strip()
        input_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if len(input_ids) < 25:
            print(item)
            remove_ids.append(item["id"])
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jasquad_metric = evaluate.load(jasquad.__file__)
        self.use_model_tok = True

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

    def doc_to_text(self, doc):
        return (
            "[題名]:"
            + doc["title"]
            + f"{self.SEP}"
            + "[問題]:"
            + doc["context"].split("[SEP]")[-1].strip()
            + f"{self.SEP}"
            + "[質問]:"
            + doc["question"]
            + f"{self.SEP}"
            + "[答え]:"
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        answer = answer_list[0]
        return answer

    def construct_requests(self, doc, ctx, **kwargs):

        args_until = dict(until=[self.SEP])
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            # continuation = rf.greedy_until(ctx, [self.SEP])
            pass
        else:
            encode_fn = self.tokenizer.encode
            if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
                encode_params = dict(add_special_tokens=False)
            else:
                encode_params = {}
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
            "id": doc["id"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }
        out = {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
        }

        return out

    def aggregation(self):
        return {
            "exact": partial(
                self._squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                self._squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }

    def _squad_metric(self, predictions, references):
        return self.jasquad_metric.compute(
            predictions=predictions, references=references
        )

    def _squad_agg(self, key, item):
        predictions, references = zip(*item)
        return self._squad_metric(predictions=predictions, references=references).get(key, 0)


@register_task("jsquad_1.1-0.2")
class JSQuADWithFintanPrompt(JSQuAD):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    PROMPT_VERSION = 0.2
    DESCRIPTION = "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。\n\n"
    SEP = "\n"

    def doc_to_text(self, doc):
        return (
            "文章:"
            + doc["context"].split("[SEP]")[-1].strip()
            + f"{self.SEP}"
            + "質問:"
            + doc["question"]
            + f"{self.SEP}"
            + "回答:"
        )
