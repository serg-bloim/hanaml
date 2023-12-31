import json
import random
import unittest
from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_something(self):
        histories = {}
        for model_path in (find_root_dir() / 'model').iterdir():
            co_path = model_path / 'custom_objects.json'
            if co_path.exists():
                with open(co_path, 'r') as f:
                    co: Dict = json.load(f)
                    hist = co.get('history')
                    if hist:
                        histories[model_path.name] = hist
        color_mapping = defaultdict(
            lambda: f"rgb({random.randint(0, 255)},{random.randint(0, 255)},{random.randint(0, 255)})")
        fig = make_subplots(rows=len(histories), cols=1, subplot_titles=list(histories.keys()), vertical_spacing=0.006)

        for i, (model, hist) in enumerate(sorted(histories.items())):
            for name, y in hist.items():
                fig.add_trace(
                    go.Scatter(y=y, name=model + "_" + name, marker_color=color_mapping[name],
                               visible='legendonly' if 'loss' in name else True),
                    row=i + 1, col=1
                )
        fig.update_layout(height=800 * len(histories), width=1500, title_text="Side By Side Subplots")
        fig.show()

    def test_compare_models_by_type(self):
        histories = {}
        for model_path in (find_root_dir() / 'model').iterdir():
            co_path = model_path / 'custom_objects.json'
            if co_path.exists():
                with open(co_path, 'r') as f:
                    co: Dict = json.load(f)
                    hist = co.get('history')
                    if hist:
                        histories[model_path.name] = hist
        color_mapping = defaultdict(
            lambda: f"rgb({random.randint(0, 255)},{random.randint(0, 255)},{random.randint(0, 255)})")
        groups = ['play', 'discard', 'clue', 'action']
        stats = ['acc', 'val_acc']
        fig = make_subplots(rows=len(histories), cols=2, subplot_titles=list(f"{g}_{s}" for g in groups for s in stats),
                            vertical_spacing=0.006)
        for i, group in enumerate(groups):
            for (model, hist) in histories.items():
                if group in model:
                    for col, stat in enumerate(stats):
                        if stat in hist:
                            fig.add_trace(
                                go.Scatter(y=hist[stat], name=model + "_" + stat, marker_color=color_mapping[model],
                                           hoverlabel=dict(namelength=-1)),
                                row=i + 1, col=col + 1
                            )
        fig.update_layout(height=800 * len(histories), width=1500, title_text="Side By Side Subplots")
        fig.show()


if __name__ == '__main__':
    unittest.main()
