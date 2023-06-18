import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import Series

from core.generate_test_cases import load_test_cases, Encoding, Field, to_dtype
from util.core import find_root_dir


def df_to_dataset(df: pd.DataFrame, shuffle=True, batch_size=5, target_name='target'):
    df = df.copy()
    labels = df.pop(target_name)
    label_lookup = tf.keras.layers.StringLookup()
    label_lookup.adapt(labels)
    numeric_labels = label_lookup(labels)
    value: Series
    df = {key: value.to_frame() for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((df, numeric_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds, label_lookup


def get_normalization_layer(field, ds):
    normalizer = tf.keras.layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = ds.map(lambda x, y: x[field.name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def create_input_pipeline(field: Field, ds: tf.data.Dataset):
    dtype = to_dtype(field.type)
    input = tf.keras.Input(shape=(1,), name=field.name, dtype=dtype)
    tf.keras.layers.InputLayer(input_shape=(1,), name=field.name, dtype=dtype)
    index = None
    encoded = None
    enc = field.get_encoding()

    if enc == Encoding.CATEGORY:
        max_tokens = field.shape or None
        if field.type == 'str':
            index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
        elif field.type == 'int':
            index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
        else:
            raise ValueError(f"Category is not supported for type ({field.type})")
        feature_ds = ds.map(lambda x, y: x[field.name])
        index.adapt(feature_ds)

        encoded_f = lambda x: tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())(index(x))
        encoded = encoded_f(input)
    elif enc == Encoding.NORMALIZE:
        encoded = tf.keras.layers.Normalization(axis=None)
        feature_ds = ds.map(lambda x, y: x[field.name])
        index.adapt(feature_ds)
    return input, encoded


class MyTestCase(unittest.TestCase):
    def test_model_v1(self):
        tc_file = next((find_root_dir() / f'data/testcases/').glob('*.tcsv'))
        with open(tc_file, 'r') as f:
            turns, fields = load_test_cases(f)
        fields_map = {f.name: f for f in fields}
        df = pd.DataFrame(turns)
        columns2remove = 'clue_number clue_color play_card'.split()
        df = df.drop(columns2remove, axis=1)
        for f in fields:
            # temporarily remove all non string fields
            if f.type != 'str':
                if f.name in df:
                    df.drop(f.name, inplace=True, axis=1)
                else:
                    pass

        df.fillna('NA', inplace=True)
        batch_size = 5
        train_ds, label_enc = df_to_dataset(df, batch_size=batch_size, target_name='action_type')
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
        tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", show_dtype=True)
        epochs = 10
        model.fit(train_ds, epochs=epochs)
        model.save(find_root_dir().joinpath(f'model/model_v1_{epochs}'))
        loss, accuracy = model.evaluate(train_ds)
        print("Accuracy", accuracy)
        predictions = model.predict(train_ds)
        predictions = [np.argmax(x) for x in predictions]
        print(predictions)
        pass

    def test_from_tensor_slices(self):
        ds = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6]))
        print(list(ds.as_numpy_iterator()))


if __name__ == '__main__':
    unittest.main()
