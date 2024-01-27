from datetime import datetime
from typing import Optional, Dict

import logging
import os
import socket

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from importlib_metadata import distribution
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ClientTracker:
    def __init__(
        self,
        experiment: str,
        trackURL: Optional[str] = None,
        run_name: Optional[str] = None,
        hparams: Optional[Dict] = dict(),
        overrides = dict(),
    ):
        self.run_id = None
        self._is_db = False

        artifact_location = None        
        if trackURL is not None:
            mlflow.set_tracking_uri(trackURL)
            # Currently only for sqlite (PostgSQL and other maybe later)
            if trackURL.startswith("sqlite"):
                artifact_location = os.path.dirname(trackURL.replace("sqlite:///", ""))
                self._is_db = True

        self._client = MlflowClient()
        try:
            if artifact_location:
                # Get len of current and use it as ID for artifacts uri
                num_exps = str(len(self._client.search_experiments(1)))
                artifact_location = Path(artifact_location).joinpath("mlruns", num_exps).as_uri()

            experiment_id = self._client.create_experiment(f"{experiment}", artifact_location)
            if self._is_db:
                self._client.create_registered_model(experiment)
        except MlflowException:
            experiment_id = self._client.get_experiment_by_name(
                f"{experiment}"
            ).experiment_id

        metadata = distribution(__name__.split(".")[0]).metadata
        if run_name is None:
            run_name += "lm_harness_" + datetime.now().strftime("%Y%m%d_%H%M")
        self.info = dict(
            exp_name=experiment,
            exp_id=experiment_id,
            run_name=run_name,
            hparams=hparams,
            overrides=overrides,
            metadata=metadata,
        )
        self.output_dir = None
        self.notifications = 0
        self.noti_arti = 0

    def __repr__(self) -> str:
        msg = "ClientTracker "
        for key, value in self.info.items():
            if key in ["overrides", "params_file", "metadata"]:
                continue
            msg += f"{key}: {value} "
        return msg

    def get_run_tags(self):
        try:
            user_name = os.getlogin()
        except OSError:
            import getpass
            user_name = getpass.getuser()
        run_tags = {
            "mlflow.runName": self.info["run_name"],
            "mlflow.source.name": self.info["metadata"]["Name"],
            "mlflow.source.type": "LOCAL",
            "mlflow.note.content": "",
            "mlflow.user": user_name,
            "mlflow.source.git.commit": self.info["metadata"]["Version"],
        }
        return run_tags

    def init_run(self):
        __local_ip = socket.gethostbyname(socket.gethostname())
        client_run = self._client.create_run(
            self.info["exp_id"], tags={
                "ip_addr": __local_ip
            }
        )
        self.run_id = client_run.info.run_id
        for key, value in self.get_run_tags().items():
            self._client.set_tag(self.run_id, key, value)

        if self.info["hparams"]:
            _hparams = yaml.dump(self.info["hparams"], default_flow_style=False)
            self._client.log_text(self.run_id, _hparams, "hyperparameters.yaml")

        if self.output_dir:
            with open(self.output_dir / "runid.mlflow", "w") as writer:
                writer.write(self.run_id)

    def init_eval(self, parent_id: Optional[str] = None, child_id: Optional[str] = None, model_tag: Optional[str] = None):
        tags = dict()
        if parent_id:
            tags["mlflow.parentRunId"] = parent_id
        tags["inference_model"] = model_tag
        if child_id is None:
            client_run = self._client.create_run(self.info["exp_id"], tags=tags)
            self.run_id = client_run.info.run_id
        else:
            self.run_id = child_id
        
        for key, value in self.get_run_tags().items():
            self._client.set_tag(self.run_id, key, value)

    def prepare_run(self, run_id: Optional[str] = None):
        if run_id:
            self.run_id = run_id
            return

        if self.output_dir:
            # Check for runid file in experiment folder
            runid_file = self.output_dir / "runid.mlflow"
            if os.path.exists(runid_file):
                with open(runid_file) as reader:
                    run_id = reader.read().replace("\n", "")
                self.run_id = run_id
                return

        # get run_id from list of experiments (always index 0)
        for run_info in self._client.search_runs(self.info["exp_id"]):
            if not hasattr(run_info, "run_id"):
                continue
            tags = self._client.get_run(run_info.run_id).data.tags
            if tags["mlflow.runName"] == self.info["run_name"]:
                self.run_id = run_info.run_id
                break

    def end_run(self):
        if self.run_id is None:
            return
        self._client.set_terminated(self.run_id)
    
    def log_stats(self, epoch: Optional[int] = None, train_stats=dict(), valid_stats=dict()):
        if self.run_id is None:
            return
        for key, value in train_stats.items():
            self._client.log_metric(self.run_id, f'train {key}', value, step=epoch)
        for key, value in valid_stats.items():
            self._client.log_metric(self.run_id, f'valid {key}', value, step=epoch)
    
    def log_stat(self, key, value, step: Optional[int] = None):
        if self.run_id is None:
            return
        try:
            self._client.log_metric(self.run_id, key, value, step=step)
        except MlflowException:
            if self.notifications < 10:
                logging.warning("The stat %s could not be stored", key)
                self.notifications += 1
            elif self.notifications == 10:
                logging.warning("The maximum warning of log stats has been reached. These warning will be disabled.")
                self.notifications += 1
        return

    def log_metrics(self, stage: str = "test", stats=dict()):
        if self.run_id is None:
            return
        for key, value in stats.items():
            self._client.log_metric(self.run_id, f"{stage} {key}", value)
    
    def log_params(self, params: dict = dict()):
        if self.run_id is None:
            return
        for key, value in params.items():
            self._client.log_param(self.run_id, key, value)
    
    def log_image(self, image, artifact_file):
        if self.run_id is None:
            return
        self._client.log_figure(self.run_id, image, artifact_file)

    def log_table(self, table, artifact_file):
        if self.run_id is None:
            return
        self._client.log_table(self.run_id, table, artifact_file)

    def log_artifact(self, artifact_file):
        if self.run_id is None:
            return
        try:
            self._client.log_artifact(self.run_id, artifact_file)
        except MlflowException:
            if self.noti_arti < 10:
                logging.warning("The artifact %s could not be stored, skipping", artifact_file)
                self.noti_arti += 1
            elif self.noti_arti == 10:
                logging.warning("The maximum warning of log artifact has been reached. These warning will be disabled.")
                self.noti_arti += 1
        return

    def log_model(self, model_path, register_model: bool = False):
        if self.run_id is None:
            return
        self._client.log_artifact(self.run_id, model_path)
        if self._is_db and register_model:
            # Server requires myql or postgre sql to create a model version
            model_name = os.path.basename(model_path)
            model_uri = "runs:/{}/{}".format(self.run_id, model_name)
            model_name = model_name.replace(".pth", "")
            self._client.create_model_version(self.info["exp_name"], model_uri, self.run_id, description=model_name)

    def log_textfile(self, filename, artifact_file):
        if self.run_id is None:
            return
        with open(filename, encoding="utf-8") as reader:
            text = reader.read()
        self._client.log_text(self.run_id, text, artifact_file)
