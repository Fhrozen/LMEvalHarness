import random

import torch

import numpy as np

import lm_eval.api
import lm_eval.tasks
import lm_eval.models
import lm_eval.api.metrics
import lm_eval.api.registry

from lm_eval.utils import (
    run_task_tests,
    get_git_commit_hash,
    simple_parse_args_string,
    eval_logger,
)

from lm_eval.evaluator import evaluate


def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :return
        Dictionary of results
    """
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(
        1234
    )  # TODO: this may affect training runs that are run with evaluation mid-run.

    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks."
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    # TODO(Fhrozen): Remove num_fewshots and move it to the Tasks python
    # or yamls. num_fewshot: 0
    if num_fewshot is None:
        num_fewshot = {}
    else:
        if isinstance(num_fewshot, list):
            if len(num_fewshot) == 0:
                num_fewshot = {}
            else:
                num_fewshot = {k: v for k, v in enumerate(num_fewshot)}
        else:
            num_fewshot = {0: num_fewshot}

    for task_idx, task_name in enumerate(task_dict.keys()):
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue

        config = task_obj._config
        if config["output_type"] == "generate_until" and gen_kwargs is not None:
            config["generation_kwargs"].update(gen_kwargs)

        _num_fewshot = num_fewshot.get(task_idx, None)
        if _num_fewshot is not None:
            if config["num_fewshot"] == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                default_num_fewshot = config["num_fewshot"]
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {_num_fewshot}"
                )

                task_obj._config["num_fewshot"] = _num_fewshot

        # TODO(Fhrozen): Try to remove this part so loading the tasks could be simplifyied
        # Maybe using trust_code in the dataset loading.

        if hasattr(task_obj, "LOAD_TOKENIZER"):
            if task_obj.LOAD_TOKENIZER:
                if isinstance(lm, lm_eval.api.model.CachingLM):
                    task_obj.tokenizer = lm.lm.tokenizer
                else:
                    task_obj.tokenizer = lm.tokenizer
        if hasattr(task_obj, "max_length"):
            task_obj.max_length = (
                lm.lm.max_length
                if isinstance(lm, lm_eval.api.model.CachingLM)
                else lm.max_length
            )
        if hasattr(task_obj, "max_gen_toks"):
            task_obj.max_gen_toks = (
                lm.lm.max_gen_toks
                if isinstance(lm, lm_eval.api.model.CachingLM)
                else lm.max_gen_toks
            )

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None
