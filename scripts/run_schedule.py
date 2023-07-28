from core.schedule import run_scheduled_tasks
from util.core import find_root_dir

run_scheduled_tasks(find_root_dir() / 'data/schedule/training.yml')
