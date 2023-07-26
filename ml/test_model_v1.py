import unittest

import numpy
import numpy as np
import pandas as pd
import progressbar
import tabulate
import tensorflow as tf

from ml.ml_util import create_input_pipeline, create_data_action, train_model, create_data_play, create_data_discard, \
    create_data_clue, load_model, TrainingData
from util.core import find_root_dir, calc_weights


class MyTestCase(unittest.TestCase):
    model_type = 'action'
    model_ver = 'v4'
    model_epochs = 5000
    model_name_suffix = '_test_unweighted'
    optimizer = 'adam'
    layers = [30, 30]
    permutate_colors = True
    batch_size = 32
    samples_pe = 32768


    def setUp(self) -> None:
        tf.get_logger().setLevel('INFO')

    def test_create_model(self, epochs=1000, save_n_epochs=100, checkpoint_n_epochs=0):
        data: TrainingData = self.create_data(permutate_colors=self.permutate_colors, batch_size=self.batch_size)
        [(train_features, label_batch)] = data.train_ds.take(1)
        all_inputs = []
        encoded_features = []
        iter = list(train_features.keys())
        iter = progressbar.progressbar(iter, prefix="Creating input pipelines")
        for feature in iter:
            field = data.fields_map[feature]
            inp, encoded = create_input_pipeline(field, data.train_ds, self.model_ver)
            all_inputs.append(inp)
            encoded_features.append(encoded)
        all_features = tf.keras.layers.concatenate(encoded_features)
        x = all_features
        for l_size in self.layers:
            x = tf.keras.layers.Dense(l_size, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(data.label_enc.vocabulary_size(), activation='softmax')(x)
        model = tf.keras.Model(all_inputs, output)
        weights = {i: 1 for i in range(1 + max(data.class_cnt.keys()))}
        weights.update(calc_weights(data.class_cnt))
        model.compile(optimizer=self.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['acc'])
        model_prefix = f"{self.model_type}_{self.model_ver}_{self.model_name_suffix}_"
        img_dir = find_root_dir() / 'model/_img'
        img_dir.mkdir(parents=True, exist_ok=True)
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", show_dtype=True,
                                  to_file=img_dir / f"{model_prefix}.png")
        train_model(model, data.train_ds, data.val_ds, data.val_ds, epochs, data.label_enc, model_prefix,
                    epoch_size=self.samples_pe // self.batch_size,
                    save_each_n_epochs=save_n_epochs,
                    checkpoint_every_n_epochs=checkpoint_n_epochs, class_weight=weights)

    def create_data(self, lbl_encoder=None, permutate_colors=False, **kwargs):
        provider = {'action': create_data_action,
                    'play': create_data_play,
                    'clue': create_data_clue,
                    'discard': create_data_discard}[self.model_type]
        return provider(self.model_ver, lbl_encoder=lbl_encoder, permutate_colors=permutate_colors, **kwargs)

    def test_improve_model(self):
        epochs = 0
        model_name = f'{self.model_type}_{self.model_ver}_{self.model_epochs}'
        model, lbl_encoder, _ = load_model(find_root_dir().joinpath(f'model/{model_name}'))
        train_ds, val_ds, test_ds, fields_map, label_enc = self.create_data(lbl_encoder)
        train_model(model, train_ds, val_ds, test_ds, epochs, label_enc, f"{self.model_type}_{self.model_ver}_",
                    save_each_n_epochs=1000,
                    starting_epoch=self.model_epochs)

    def test_run_existing_models(self):
        output = []
        model_ver = "v4"
        for model_dir in find_root_dir().joinpath(f'model').glob(f"{self.model_type}_{model_ver}*"):
            model, enc, _ = load_model(model_dir)
            train_ds, val_ds, test_ds, fields_map, label_enc = self.create_data(enc)
            loss, accuracy, *_ = model.evaluate(train_ds)
            output.append([model_dir.name, loss, accuracy])
        print(tabulate.tabulate(output, headers="Name Loss Accuracy".split()))

    def test_predict_existing_model(self):
        model_ver = "v4"
        epoch = 1600
        model_dir = find_root_dir() / f'model/action_{model_ver}_{epoch}'
        model, encoder, custom_objs = load_model(model_dir)
        test_ds: tf.data.Dataset
        train_ds, val_ds, test_ds, fields_map, label_enc = self.create_data()
        test_ds_unbatched = test_ds.unbatch()
        vocabulary = label_enc.get_vocabulary()
        actual_labels = [vocabulary[row[1].numpy()] for row in test_ds_unbatched]
        predictions = model.predict(test_ds)
        loss, accuracy, *_ = model.evaluate(test_ds.rebatch(1))
        print(f"Evaluation ({len(actual_labels)}). Loss: {loss}, accuracy: {accuracy}")
        results = [vocabulary[numpy.argmax(p)] for p in predictions]
        certainty = [max(p) for p in predictions]
        match = ['+' if a == p else '' for a, p in zip(actual_labels, results)]
        print(tabulate.tabulate(zip(actual_labels, results, certainty, match),
                                headers=['actual', 'prediction', 'certainty', 'match']))
        pass

    def test_predict_action(self):
        model: tf.keras.models.Model
        model, enc, custom = load_model(find_root_dir() / 'model/action_v4_test_unweighted_100')
        train_ds, val_ds, test_ds, fields_map, label_enc, *_ = create_data_action(self.model_ver)

        def data_distribution(ds):
            res = model.predict(ds)
            analyze = pd.DataFrame()
            analyze['true_lbl_num'] = [row[1].numpy() for row in ds.unbatch()]
            analyze['true_lbl'] = analyze['true_lbl_num'].map(enc.get_vocabulary().__getitem__)
            analyze['actual_data'] = [x for x in res]
            analyze['argmax'] = analyze['actual_data'].map(np.argmax)
            analyze['certainty'] = analyze['actual_data'].map(np.max)
            analyze['prediction'] = analyze['argmax'].map(enc.get_vocabulary().__getitem__)
            print(analyze.true_lbl.value_counts())
            col1 = analyze.true_lbl
            col2 = analyze.prediction
            uniq = col1.unique().tolist()
            sss = [(v, col1 == v) for v in uniq] + [('all', col1.map(lambda x: True))]
            sss.sort()
            data = []
            for name, ss in sss:
                l = sum(ss)
                acc = sum(col1[ss] == col2[ss]) / l * 100
                data.append([name, l, 100 * l / len(col1), acc] + [sum(col2[ss] == x) / l * 100 for x in uniq])
            print(tabulate.tabulate(data, headers=('name length % acc' + ''.join(f" {x}" for x in uniq)).split()))

        print(f"Validation data:")
        data_distribution(val_ds)

        print(f"\n\nTrain data:")
        data_distribution(train_ds)


if __name__ == '__main__':
    unittest.main()
