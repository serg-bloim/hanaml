import sys
import uuid
from typing import Dict, NamedTuple, List

import yaml

from ml.test_model_v1 import MyTestCase


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

    def set_state(self, state):
        self._cfg['state'] = state


class TrainModelTask(ScheduledTask):

    def update_config(self, cfg: Dict):
        class TrainModelTaskConfig(NamedTuple):
            id: str
            type: str
            epochs: int
            target: str
            state: str = 'new'
            layers: List[int] = [30, 30]
            data_ver: str = 'v4'
            checkpoint_n_epochs: int = 100
            save_n_epochs: int = 1000

        self._cfg_parsed = TrainModelTaskConfig(**cfg)
        return super().update_config(cfg)

    def _exec(self):
        test = MyTestCase()
        test.setUp()
        cfg = self._cfg_parsed
        test.model_type = cfg.target
        test.model_ver = cfg.data_ver
        test.model_name_suffix = ''.join(f"_{x}" for x in cfg.layers)
        print(f"\n\nRunning task {self.id()}")
        print(f"Training model type {cfg.target} for {cfg.epochs} epochs with layers config: {test.model_name_suffix}\n")
        test.test_create_model(cfg.epochs, save_n_epochs=cfg.save_n_epochs, checkpoint_n_epochs=cfg.checkpoint_n_epochs)


class Schedule:
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.tasks: Dict[str, ScheduledTask] = {}

    def update(self):
        with open(self.filepath, 'r') as f:
            data = yaml.safe_load_all(f)
            need_save = False
            for task_cfg in data:
                if 'id' not in task_cfg:
                    old_cfg = task_cfg
                    task_cfg = dict(id=str(uuid.uuid4()))
                    task_cfg.update(old_cfg)
                id = task_cfg['id']
                if id not in self.tasks:
                    self.tasks[id] = self.create_task(task_cfg)
                    need_save = True
                else:
                    need_save = need_save or self.tasks[id].update_config(task_cfg)
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

    def has_incomplete(self):
        return any(not t.completed() for t in self.tasks.values())

    def get_next_incomplete(self):
        return next(t for t in self.tasks.values() if not (t.completed() or t.is_failed()))

    def run_task(self, task):
        task.set_state('running')
        self.save()
        task.run()
        task.set_state('failed' if task.is_failed() else 'complete')
        self.save()
        if task.is_failed():
            print(f"Task {task.id()} failed")
            print(task.get_error(), file=sys.stderr)
        else:
            print(f"Task {task.id()} is complete succesfully")


def run_scheduled_tasks(schedule_filepath):
    schedule = Schedule(schedule_filepath)
    while True:
        schedule.update()
        if not schedule.has_incomplete():
            break
        task = schedule.get_next_incomplete()
        schedule.run_task(task)
        schedule.save()
