import os
import random
import string
import traceback
import uuid
from pathlib import Path
from typing import Dict, NamedTuple, List

import filelock
import yaml
from filelock import FileLock

from ml.test_model_v1 import MyTestCase
from util.core import find_root_dir


class CleanUpFileLock(FileLock):
    def __init__(self, lock_path, timeout=-1):
        super().__init__(lock_path, timeout=timeout)
        self.lock_path = lock_path

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.lock_path)
        super().__exit__(exc_type, exc_value, traceback)


class ScheduledTask:

    def __init__(self):
        self.__err = None
        self._id = None
        self._cfg = {}

    def update_config(self, cfg: Dict) -> bool:
        old = self._cfg
        self._cfg = cfg
        return self._cfg == old

    def get_config(self):
        return self._cfg

    def completed(self):
        return self._cfg.get('state') == 'complete'

    def is_failed(self):
        return self.__err is not None

    def get_error(self):
        return self.__err

    def run(self):
        try:
            self._exec()
            self.set_state('complete')
        except Exception as e:
            self.__err = e
            self.set_state('failed')
            return False

    def id(self):
        return self._cfg['id']

    def _exec(self):
        raise NotImplementedError()

    def get_state(self):
        return self._cfg.get('state', 'new')

    def set_state(self, state):
        self._cfg['state'] = state


class Schedule:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.tasks: Dict[str, ScheduledTask] = {}

    def update(self, save_tasks=None):
        save_tasks_ids = [t.id() for t in save_tasks or []]
        with CleanUpFileLock(self.filepath.with_suffix(self.filepath.suffix + ".lock")):
            with open(self.filepath, 'r') as f:
                data = yaml.safe_load_all(f)
                need_save = False
                for task_cfg in data:
                    if task_cfg is None:
                        continue
                    if 'id' not in task_cfg:
                        old_cfg = task_cfg
                        task_cfg = dict(id=str(uuid.uuid4()))
                        task_cfg.update(old_cfg)
                    id = task_cfg['id']
                    if id not in self.tasks:
                        self.tasks[id] = self.create_task(task_cfg)
                        need_save = True
                    if id not in save_tasks_ids:
                        need_save = need_save or self.tasks[id].update_config(task_cfg)
            if save_tasks is not None:
                need_save = True
            if need_save:
                self.save()

    def create_task(self, cfg):
        clazz = {'train': TrainModelTask}[cfg['type']]
        inst = clazz()
        inst.update_config(cfg)
        return inst

    def save(self):
        with open(self.filepath, 'w') as f:
            yaml.safe_dump_all([d.get_config() for d in self.tasks.values()], f, sort_keys=False)

    def run_task(self, task):
        task.set_state('running')
        self.update(save_tasks=[task])
        task.run()
        task.set_state('failed' if task.is_failed() else 'complete')
        self.update(save_tasks=[task])
        if task.is_failed():
            print(f"Task {task.id()} failed")
            traceback.print_exception(task.get_error())
        else:
            print(f"Task {task.id()} is complete succesfully")

    def stream_incomplete_tasks(self):
        while True:
            self.update()
            for task in self.tasks.values():
                if task.get_state() in ['new', 'running', None]:
                    try:
                        with self.__try_aquire_task_lock(task):
                            if task.get_state() in ['new', 'running', None]:
                                yield task
                    except filelock.Timeout:
                        pass
            else:
                break

    def __try_aquire_task_lock(self, task):
        lock_path = find_root_dir() / f'data/schedule/.task_locks/{task.id()}.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        return CleanUpFileLock(lock_path, 0)


class TrainModelTask(ScheduledTask):

    def update_config(self, cfg: Dict):
        class TrainModelTaskConfig(NamedTuple):
            id: str
            type: str
            epochs: int
            target: str
            permutate_colors: bool = True
            optimizer: str = 'adam'
            batch_size: int = 5
            state: str = 'new'
            prefix: str = '_scheduled'
            layers: List[int] = [30, 30]
            data_ver: str = 'v4'
            checkpoint_n_epochs: int = 100
            save_n_epochs: int = 1000

        self._cfg_parsed = TrainModelTaskConfig(**cfg)
        return super().update_config(cfg)

    def _exec(self):
        import tensorflow as tf
        test = MyTestCase()
        test.setUp()
        cfg = self._cfg_parsed
        test.model_type = cfg.target
        test.model_ver = cfg.data_ver
        test.permutate_colors = cfg.permutate_colors
        opt = cfg.optimizer
        opt_str = cfg.optimizer
        test.batch_size = cfg.batch_size
        if isinstance(cfg.optimizer, dict):
            opt = tf.keras.optimizers.deserialize(cfg.optimizer)
            opt_str = cfg.optimizer['class_name']
        test.optimizer = opt
        rnd = ''.join(random.choices(string.ascii_lowercase, k=5))
        test.model_name_suffix = cfg.prefix + '_' + opt_str + ''.join(f"_{x}" for x in cfg.layers) + '_' + rnd
        print(f"\n\nRunning task {self.id()}")
        print(yaml.safe_dump(self.get_config()))
        print(
            f"Training model type {cfg.target} for {cfg.epochs} epochs with layers config: {test.model_name_suffix}\n")
        test.test_create_model(cfg.epochs, save_n_epochs=cfg.save_n_epochs, checkpoint_n_epochs=cfg.checkpoint_n_epochs)


def run_scheduled_tasks(schedule_filepath):
    schedule = Schedule(schedule_filepath)
    for task in schedule.stream_incomplete_tasks():
        schedule.run_task(task)
