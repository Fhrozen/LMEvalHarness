import datasets

from lm_eval.api.task import MultipleChoiceTask as LMMultiChoiceTask
from lm_eval.api.task import TaskConfig
from lm_eval.api.task import Task as LMTask
from lm_eval.filters import build_filter_ensemble


class Task(LMTask):
    num_fewshot = 0
    fewshot_delimiter = "\n\n"
    dataset_kwargs = {}
    description = ""

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None) -> None:
        # Custom Config
        _config = dict(
            num_fewshot = self.num_fewshot,
            fewshot_delimiter = self.fewshot_delimiter,
            description = self.description,
            dataset_kwargs = dict(
                trust_remote_code=True,
                **self.dataset_kwargs
            ) 
        )
        if config is not None:
            _config.update(**config)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None

        self._config = TaskConfig({**_config})
        self.download(self._config.dataset_kwargs)

        self._filters = [build_filter_ensemble("none", [["take_first", None]])]

    def download(self, dataset_kwargs=None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )


class MultipleChoiceTask(LMMultiChoiceTask):
    num_fewshot = 0
    fewshot_delimiter = "\n\n"
    dataset_kwargs = {}
    description = ""

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None) -> None:
        # Custom Config
        _config = dict(
            num_fewshot = self.num_fewshot,
            fewshot_delimiter = self.fewshot_delimiter,
            description = self.description,
            dataset_kwargs = dict(
                trust_remote_code=True,
                **self.dataset_kwargs
            ) 
        )

        if config is not None:
            _config.update(**config)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None

        self._config = TaskConfig({**_config})
        self.download(self._config.dataset_kwargs)

        self._filters = [build_filter_ensemble("none", [["take_first", None]])]

    def download(self, dataset_kwargs=None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )
