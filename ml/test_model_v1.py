import json
import unittest
from typing import Tuple, Any

import numpy
import tabulate
import tensorflow as tf
from keras.layers import StringLookup
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

from ml.ml_util import create_input_pipeline, create_data_action, train_model, create_data_play, create_data_discard, \
    create_data_clue, load_model
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    model_type = 'action'
    model_ver = 'v3'
    model_epochs = 5000

    def setUp(self) -> None:
        tf.get_logger().setLevel('INFO')

    def test_create_model(self):
        epochs = 5000

        train_ds, test_ds, fields_map, label_enc = self.create_data()
        [(train_features, label_batch)] = train_ds.take(1)
        print('Every feature:', list(train_features.keys()))
        print('A batch of turns:', train_features['active_card_1_clue_color'])
        print('A batch of targets:', label_batch)

        all_inputs = []
        encoded_features = []
        for feature in train_features.keys():
            field = fields_map[feature]
            inp, encoded = create_input_pipeline(field, train_ds)
            all_inputs.append(inp)
            encoded_features.append(encoded)
        all_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(20, activation="relu")(all_features)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(label_enc.vocabulary_size(), activation='softmax')(x)
        model = tf.keras.Model(all_inputs, output)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['acc'])
        model_prefix = f"{self.model_type}_{self.model_ver}_"
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", show_dtype=True, to_file=model_prefix + ".png")
        train_model(model, train_ds, test_ds, epochs, label_enc, model_prefix, save_each_n_epochs=1000)

    def create_data(self, lbl_encoder:StringLookup = None) -> Tuple[DatasetV1Adapter, DatasetV1Adapter, object, StringLookup]:
        provider = {'action': create_data_action,
                    'play': create_data_play,
                    'clue': create_data_clue,
                    'discard': create_data_discard}[self.model_type]
        return provider(self.model_ver, lbl_encoder)

    def test_improve_model(self):
        epochs = 0
        model_name = f'{self.model_type}_{self.model_ver}_{self.model_epochs}'
        model, lbl_encoder = load_model(find_root_dir().joinpath(f'model/{model_name}'))
        train_ds, test_ds, fields_map, label_enc = self.create_data(lbl_encoder)
        train_model(model, train_ds, test_ds, epochs, label_enc, f"{self.model_type}_{self.model_ver}_",
                    save_each_n_epochs=1000,
                    starting_epoch=self.model_epochs)

    def test_run_existing_models(self):
        output = []
        model_ver = "v3"
        for model_dir in find_root_dir().joinpath(f'model').glob(f"{self.model_type}_{model_ver}*"):
            model = tf.keras.models.load_model(model_dir)
            train_ds, test_ds, fields_map, label_enc = self.create_data()
            loss, accuracy, *_ = model.evaluate(test_ds)
            output.append([model_dir.name, loss, accuracy])
        print(tabulate.tabulate(output, headers="Name Loss Accuracy".split()))

    def test_predict_existing_model(self):
        model_ver = "v3"
        epoch = 17000
        model_dir = find_root_dir() / f'model/discard_{model_ver}_{epoch}'
        model: tf.keras.Model = tf.keras.models.load_model(model_dir)
        test_ds: tf.data.Dataset
        train_ds, test_ds, fields_map, label_enc = self.create_data()
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
        class VocabularyLayer(tf.keras.layers.Layer):

            def __init__(self, vocab, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
                super().__init__(trainable, name, dtype, dynamic, **kwargs)
                self.vocab = vocab

            def call(self, inputs, *args, **kwargs):
                return tf.nest.map_structure(lambda x: {'prediction': x, 'vocabulary': self.vocab}, inputs)

        model_dir = find_root_dir() / f'model/action_v2_7000'
        model: tf.keras.Model = tf.keras.models.load_model(model_dir)

        model2 = tf.keras.Model(model.inputs, VocabularyLayer('abc def hij klm'.split())(model.output))
        input: dict[Any, Any] = {}
        with open(find_root_dir() / 'data/input.json', 'r') as f:
            input = json.load(f)
        print(input)
        data = tf.data.Dataset.from_tensor_slices({f: [v] for f, v in input.items()})
        prediction = model.predict(data.batch(1))
        print(prediction)
        prediction = model2.predict(data.batch(1))
        print(prediction)

        pass

    def test_from_tensor_slices(self):
        ds = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6]))
        print(list(ds.as_numpy_iterator()))


if __name__ == '__main__':
    unittest.main()
