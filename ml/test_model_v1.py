import unittest

import tabulate
import tensorflow as tf

from ml.ml_util import create_input_pipeline, create_data, train_model
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_create_model(self):
        model_ver = 'v2'
        epochs = 5000

        train_ds, test_ds, fields_map, label_enc = create_data(model_ver)
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
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(label_enc.vocabulary_size(), activation='softmax')(x)
        model = tf.keras.Model(all_inputs, output)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        train_model(model, train_ds, test_ds, epochs, label_enc, f"model_{model_ver}_{epochs}")

    def test_improve_model(self):
        model_ver = 'v2'
        model_epochs = 2000
        epochs = 8000
        model_name = f'model_{model_ver}_{model_epochs}'
        model = tf.keras.models.load_model(find_root_dir().joinpath(f'model/{model_name}'))
        train_ds, test_ds, fields_map, label_enc = create_data(model_ver)
        train_model(model, train_ds, test_ds, epochs, label_enc, f"model_{model_ver}_", save_each_n_epochs=1000,
                    starting_epoch=model_epochs)

    def test_run_existing_models(self):
        output = []
        model_ver = "v1"
        for model_dir in find_root_dir().joinpath(f'model').glob(f"model_{model_ver}*"):
            model = tf.keras.models.load_model(model_dir)
            train_data, test_data, *_ = create_data(model_ver)
            loss, accuracy = model.evaluate(test_data)
            output.append([model_dir.name, loss, accuracy])
        print(tabulate.tabulate(output, headers="Name Loss Accuracy".split()))

    def test_from_tensor_slices(self):
        ds = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6]))
        print(list(ds.as_numpy_iterator()))


if __name__ == '__main__':
    unittest.main()
