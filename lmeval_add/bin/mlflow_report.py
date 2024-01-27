import argparse
import glob
import logging
import json
import os
import sys

import numpy as np

from lm_eval import utils
from lmeval_add.utils.mlflow import ClientTracker


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--experiment_name",
        default="lm_harness",
        help="",
        type=str,
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="",
        type=str,
    )
    parser.add_argument(
        "--track_url",
        default=None,
        help="",
        type=str,
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="",
        type=str,
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--config_details",
        default=None,
        help="",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_eval_args()
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))

    conf = {}
    if args.config_details is not None:
        conf = [x.split("=") for x in args.config_details.split(",") if "=" in x]
        conf = {k: v for k,v in conf}

    if args.results_dir is None:
        eval_logger.info("--results_dir is None, exiting.")
        sys.exit(0)

    results_file = os.path.join(args.results_dir, "results.json")
    if not os.path.exists(results_file):
        eval_logger.info("results.log does not exists in the result_dir, exiting.")
        sys.exit(0)

    client_ml = ClientTracker(
        experiment=args.experiment_name,
        trackURL=args.track_url,
        run_name=args.run_name,
        hparams=conf,
    )
    client_ml.prepare_run()
    if client_ml.run_id is None:
        client_ml.init_run()

    with open(results_file, encoding="utf-8") as reader:
        results_data = json.load(reader)
    all_scores = dict()
    params = dict()
    for key, values in results_data.items():
        if key == "results":
            for task, scores in values.items():
                for scr, val in scores.items():
                   scr = scr.split(",")[0]
                   if isinstance(val, int) or isinstance(val, float):
                       _key = f"{task}/{scr}"
                       all_scores[_key] = float(val)
        elif key == "config":
            for k, val in values.items():
                if k in ["model_args", "gen_kwargs"]:
                    continue
                params[k] = val

    if len(all_scores) > 0:
        client_ml.log_metrics("test", all_scores)
    if len(params) > 0:
        client_ml.log_params(params)

    details_files = glob.glob(os.path.join(args.results_dir, "*.jsonl"))
    for fn in details_files:
        # Assuming common filename of each jsonl
        task_name = os.path.basename(fn).replace(".jsonl", "").split(",")[-1]
        with open(fn, encoding="utf-8") as reader:
            details = json.load(reader)

        inputs = []
        target = []
        outputs = []
        scores = dict()

        for docs in details:
            for key, value in docs.items():
                if key == "doc":
                    for k, _val in value.items():
                        if k in ["question", "sentence", "goal", "hypothesis", "query"]:
                            break
                        _val = None
                    if _val is None:
                        _val = ""
                    inputs.append(_val)
                elif key == "target":
                    target.append(str(value))
                elif key == "filtered_resps":
                    if isinstance(value[0], list) and len(value[0]) == 2:
                        # For multichoice option
                        _val = np.array([x[0] for x in value])
                        _val = np.argmax(_val)
                    else:
                        # For single choice
                        _val = value[0]
                    outputs.append(str(_val))
                elif key in ["acc"]:
                    if key not in scores:
                        scores[key] = list()
                    scores[key].append(value)

        _table = dict(
            inputs = inputs,
            outputs = outputs,
            target = target,
        )
        _table.update(**scores)
        client_ml.log_table(_table, f"{task_name}_results.json")
    client_ml.end_run()
    sys.exit(0)
