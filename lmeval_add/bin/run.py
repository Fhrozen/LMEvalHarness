# from lm_eval.__main__ import cli_evaluate
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

from lm_eval import utils
from lm_eval.__main__ import parse_eval_args, _handle_non_serializable
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table

from lmeval_add.evaluator import simple_evaluate
from lmeval_add.tasks.japanese import * 


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks(args.verbosity)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(sorted(ALL_TASKS)))
        )
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks_list
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", args.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
