import pathlib
from typing import Any, Callable

import numpy as np
import pandas as pd
import progressbar
import tabulate
import tensorflow as tf
from pandas import Series

from core.generate_test_cases import load_test_cases, Field, to_dtype, Encoding
from util.core import find_root_dir


def create_data(ver):
    all_turns = []
    all_fields = []
    for dataset in ['train', 'test']:
        for fn in (find_root_dir() / f'data/testcases/' / ver).glob(dataset + '*.tcsv'):
            with open(fn, 'r') as f:
                turns, fields = load_test_cases(f)
                for t in turns:
                    t['dataset'] = dataset
                all_turns += turns
                all_fields.append(fields)

    fields_map = {f.name: f for f in fields}
    columns2remove = 'clue_number clue_color play_card'.split()
    df = pd.DataFrame(all_turns)
    df = df.drop(columns2remove, axis=1)
    df.fillna('NA', inplace=True)
    batch_size = 5
    dataset_lbl = df.pop('dataset')
    df_train = df[dataset_lbl == 'train']
    df_test = df[dataset_lbl == 'test']
    train_ds, label_enc = df_to_dataset(df_train, batch_size=batch_size, target_name='action_type', shuffle=True)
    test_ds, label_enc = df_to_dataset(df_test, batch_size=batch_size, target_name='action_type',
                                       target_encoder=label_enc)
    return train_ds, test_ds, fields_map, label_enc


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
    encoded = input
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
    elif enc == Encoding.AS_IS and field.type == 'int':
        encoded = tf.cast(input, dtype='float32')

    return input, encoded


def df_to_dataset(df: pd.DataFrame, shuffle=True, batch_size=5, target_name='target', target_encoder=None):
    df = df.copy()
    labels = df.pop(target_name)
    if target_encoder is None:
        target_encoder = tf.keras.layers.StringLookup()
        target_encoder.adapt(labels)
    numeric_labels = target_encoder(labels)
    value: Series
    df = {key: value.to_frame() for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((df, numeric_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds, target_encoder


def train_model(model: tf.keras.Model, train_ds, test_ds, epochs, label_enc, save_name, save_each_n_epochs=None,
                callbacks=None, starting_epoch=0):
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", show_dtype=True)
    print(f"Start training. Datasize: {len(train_ds)}/{len(test_ds)}")

    bar = progressbar.ProgressBar(min_value=starting_epoch, max_value=starting_epoch + epochs,
                                  suffix=' loss: {variables.loss}, accuracy:{variables.accuracy}',
                                  variables={'loss': '', 'accuracy': ''})
    all_callbacks = [EpochsProgressBar(bar, starting_epoch=starting_epoch)]
    model_naming = lambda e: find_root_dir() / f'model/{save_name}{e}'
    if save_each_n_epochs:
        all_callbacks.append(SaveEveryNEpochs(save_each_n_epochs, model_naming,
                                              starting_epoch=starting_epoch))
    if callbacks:
        all_callbacks += callbacks

    model.fit(train_ds, epochs=epochs, verbose=0, callbacks=all_callbacks)
    model.save(model_naming(starting_epoch + epochs))
    loss, accuracy = model.evaluate(test_ds)
    print("Test evaluation accuracy", accuracy)
    predictions = model.predict(test_ds)
    predictions = [np.argmax(x) for x in predictions]
    print(predictions)
    labels = [l for f, l in test_ds.unbatch().as_numpy_iterator()]
    lvocab = label_enc.get_vocabulary()
    data = [[lvocab[x] for x in [l, p]] for l, p in zip(labels, predictions)]
    print(tabulate.tabulate(data, headers='actual predicted'.split()))


class EpochsProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, bar: progressbar.ProgressBar, starting_epoch=0):
        super().__init__()
        self.bar = bar
        self.starting_epoch = starting_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.bar.update(epoch + self.starting_epoch, **{k: str(v)[:6] for k, v in logs.items()})


class SaveEveryNEpochs(tf.keras.callbacks.Callback):

    def __init__(self, period: int, naming: Callable[[Any], pathlib.Path | str], starting_epoch=0):
        super().__init__()
        self.period = period
        self.naming = naming
        self.starting_epoch = starting_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            epoch += self.starting_epoch
            if epoch % self.period == 0:
                self.model.save(self.naming(epoch))