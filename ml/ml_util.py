import collections
import itertools
import json
import pathlib
import shutil
from json import JSONEncoder
from pathlib import Path
from typing import Any, Callable, List, NamedTuple
from typing import Dict

import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf
from keras.layers import StringLookup, IntegerLookup
from pandas import Series
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter, DatasetV1, DatasetV2

from core.generate_test_cases import load_test_cases, Field, to_dtype, Encoding
from util.core import find_root_dir


class ModelResponse:

    def __init__(self, prediction: List[float], decoder: Callable[[int], str]) -> None:
        super().__init__()
        self.prediction = prediction
        self.decoder = decoder

    def top_result(self):
        top_ind = np.argmax(self.prediction)
        return self.decoder(top_ind)


def fill_na(v, replacement):
    return replacement if v is None else v


class ModelContainer:

    def __init__(self, tf_model: tf.keras.Model, result_decoder: Callable[[int], str]) -> None:
        self.result_decoder = result_decoder
        self.model = tf_model

    def request(self, inputs: Dict) -> ModelResponse:
        ds = tf.data.Dataset.from_tensor_slices({f: [fill_na(v, 'NA')] for f, v in inputs.items()}).batch(1)
        try:
            prediction = self.model.predict(ds)
            return ModelResponse(prediction, self.result_decoder)
        except:
            pass
        pass


def create_custom_data(df, target_column, lbl_encoder: StringLookup = None, permutate_colors=False, batch_size=5):
    columns2remove = 'action_type clue_number clue_color play_card'.split()

    try:
        columns2remove.remove(target_column)
    except:
        pass
    df = df.drop(columns2remove, axis=1)
    df.fillna('NA', inplace=True)
    dataset_lbl = df.pop('dataset')
    df_train = df[dataset_lbl == 'train']
    class_cnt = collections.Counter(df_train[target_column])
    if permutate_colors:
        class ColoredFields:
            def __init__(self, color) -> None:
                self.color = color
                self.stack = []
                self.avail_1 = []
                self.avail_2 = []
                self.avail_3 = []
                self.avail_4 = []
                self.avail_5 = []

        colornumber_field_names = ['clue_val']
        replace_field_names = [tpl % (n + 1) for n in range(5) for tpl in
                               ['active_card_%s_clue_color', 'opponent_card_%s_color', 'opponent_card_%s_clue_color']]
        if 'clue_color' in df_train:
            replace_field_names.append('clue_color')
        copy_field_names = ['clues', 'turn', 'action_type', 'clue_number', 'play_card']
        for n in range(5):
            copy_field_names.append(f"active_card_{n + 1}_clue_number")
            copy_field_names.append(f"opponent_card_{n + 1}_clue_number")
            copy_field_names.append(f"opponent_card_{n + 1}_number")
        copy_field_names = [fn for fn in copy_field_names if fn in df_train]
        colornumber_field_names = [x for x in colornumber_field_names if x in df_train]

        def permutate(rec):
            colors = ['r', 'g', 'y', 'b', 'w']
            replace_fields: Dict[str, List] = {f: [] for f in replace_field_names}
            colornumber_fields: Dict[str, List] = {f: [] for f in colornumber_field_names}
            copy_fields: Dict[str, List] = {f: [] for f in copy_field_names}
            shuffle_fields = [ColoredFields(c) for c in colors]
            perm: List[ColoredFields]
            for perm in itertools.permutations(shuffle_fields):
                mapping: Dict[str, ColoredFields] = {c: p for c, p in zip(colors, perm)}

                for name, data in copy_fields.items():
                    val = rec.__getattribute__(name)
                    data.append(val)
                for name, data in replace_fields.items():
                    val = rec.__getattribute__(name)
                    if val != 'NA':
                        val = mapping[val].color
                    data.append(val)
                for name, data in colornumber_fields.items():
                    val = rec.__getattribute__(name)
                    if val in mapping:
                        val = mapping[val].color
                    data.append(val)
                for c, f in mapping.items():
                    f.stack.append(rec.__getattribute__('stack_' + c))
                    f.avail_1.append(rec.__getattribute__(f'avail_{c}1'))
                    f.avail_2.append(rec.__getattribute__(f'avail_{c}2'))
                    f.avail_3.append(rec.__getattribute__(f'avail_{c}3'))
                    f.avail_4.append(rec.__getattribute__(f'avail_{c}4'))
                    f.avail_5.append(rec.__getattribute__(f'avail_{c}5'))
            data = copy_fields
            data.update(replace_fields)
            data.update(colornumber_fields)
            for f in shuffle_fields:
                data[f'stack_{f.color}'] = f.stack
                data[f'avail_{f.color}1'] = f.avail_1
                data[f'avail_{f.color}2'] = f.avail_2
                data[f'avail_{f.color}3'] = f.avail_3
                data[f'avail_{f.color}4'] = f.avail_4
                data[f'avail_{f.color}5'] = f.avail_5
            return pd.DataFrame(data)

        df_train = pd.concat([permutate(rec) for rec in
                              progressbar.progressbar(df_train.itertuples(), max_value=len(df_train),
                                                      prefix="color permutations")], axis=0, copy=False)
        df_train = df_train.iloc[np.random.permutation(len(df_train))]
        print(f"Done concat {len(df_train)}")
    df_test = df[dataset_lbl == 'test']
    df_val = df[dataset_lbl == 'val']
    train_ds: DatasetV1Adapter
    train_ds, lbl_encoder = df_to_dataset(df_train, batch_size=batch_size, target_name=target_column,
                                          target_encoder=lbl_encoder)
    val_ds, lbl_encoder = df_to_dataset(df_val, batch_size=batch_size, target_name=target_column, shuffle=False,
                                        target_encoder=lbl_encoder)
    test_ds, lbl_encoder = df_to_dataset(df_test, batch_size=batch_size, target_name=target_column, shuffle=False,
                                         target_encoder=lbl_encoder)
    return TrainingData(train_ds, val_ds, test_ds, lbl_encoder,
                        class_cnt={lbl_encoder(k).numpy().astype(int): v for k, v in class_cnt.items()})


def load_dataframe(ver):
    all_turns = []
    all_fields = []
    for dataset in ['train', 'test', 'val']:
        for fn in (find_root_dir() / f'data/testcases/' / ver).glob(dataset + '*.tcsv'):
            with open(fn, 'r') as f:
                turns, fields = load_test_cases(f)
                for t in turns:
                    t['dataset'] = dataset
                all_turns += turns
                all_fields.append(fields)
    fields_map = {f.name: f for f in fields}
    df = pd.DataFrame(all_turns)
    return df, fields_map


class TrainingData(NamedTuple):
    train_ds: DatasetV1 | DatasetV2
    val_ds: DatasetV1 | DatasetV2
    test_ds: DatasetV1 | DatasetV2
    label_enc: StringLookup | IntegerLookup
    fields_map: Dict[str, Field] = None,
    class_cnt: Dict[int, int] = None


def create_data_action(ver, **kwargs):
    target_column = 'action_type'
    df, fields_map = load_dataframe(ver)
    data: TrainingData = create_custom_data(df, target_column, **kwargs)
    return data._replace(fields_map=fields_map)


def create_data_play(ver, **kwargs):
    target_column = 'play_card'
    df, fields_map = load_dataframe(ver)
    df = df[df['action_type'] == 'play']
    data = create_custom_data(df, target_column, **kwargs)
    return data._replace(fields_map=fields_map)


def create_data_clue(ver, **kwargs):
    target_column = 'clue_val'
    df, fields_map = load_dataframe(ver)
    df['clue_val'] = df.clue_number.fillna(df.clue_color)
    df = df[df['action_type'] == 'clue']
    data = create_custom_data(df, target_column, **kwargs)
    return data._replace(fields_map=fields_map)


def create_data_discard(ver, **kwargs):
    target_column = 'play_card'
    df, fields_map = load_dataframe(ver)
    df = df[df['action_type'] == 'discard']
    data = create_custom_data(df, target_column, **kwargs)
    return data._replace(fields_map=fields_map)


def get_normalization_layer(field, ds):
    normalizer = tf.keras.layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = ds.map(lambda x, y: x[field.name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def create_input_pipeline(field: Field, ds: tf.data.Dataset, data_ver):
    dtype = to_dtype(field.type)
    input = tf.keras.Input(shape=(1,), name=field.name, dtype=dtype)
    tf.keras.layers.InputLayer(input_shape=(1,), name=field.name, dtype=dtype)
    encoded = input
    enc = field.get_encoding()

    if enc == Encoding.CATEGORY:
        max_tokens = field.shape or None
        if field.type == 'str':
            index = tf.keras.layers.StringLookup(max_tokens=max_tokens, vocabulary=list(field.vocabulary))
        elif field.type == 'int':
            index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens, vocabulary=[int(x) for x in field.vocabulary])
        else:
            raise ValueError(f"Category is not supported for type ({field.type})")

        encoded_f = lambda x: tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())(index(x))
        encoded = encoded_f(input)
    elif enc == Encoding.NORMALIZE:
        encoded = tf.keras.layers.Normalization(axis=None)
        feature_ds = ds.map(lambda x, y: x[field.name])
        encoded.adapt(feature_ds)
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


def train_model(model: tf.keras.Model, train_ds, val_ds, test_ds, epochs, label_enc, save_name, save_each_n_epochs=None,
                callbacks=None, starting_epoch=0, checkpoint_every_n_epochs=100, epoch_size=1000, class_weight=None):
    print(f"Start training. Datasize: {len(train_ds)}/{len(test_ds)}")

    bar = progressbar.ProgressBar(max_value=epochs,
                                  suffix=' loss: {variables.loss}, accuracy:{variables.acc} val_loss: {variables.val_loss}, val_acc:{variables.val_acc}',
                                  variables={'loss': '', 'acc': '', 'val_loss': '', 'val_acc': ''},
                                  term_width=120
                                  )
    all_callbacks = [EpochsProgressBar(bar)]
    model_naming = lambda e: find_root_dir() / f'model/{save_name}{e}'
    cp_naming = lambda e: find_root_dir() / f'model/{save_name}{e}_cp'
    cp_saver = None
    if checkpoint_every_n_epochs:
        cp_saver = SaveEveryNEpochs(checkpoint_every_n_epochs, cp_naming, label_enc, starting_epoch=starting_epoch,
                                    remove_prev=True)
        all_callbacks.append(cp_saver)
    if save_each_n_epochs:
        all_callbacks.append(SaveEveryNEpochs(save_each_n_epochs, model_naming, label_enc,
                                              starting_epoch=starting_epoch, remove_prev=False, backup=cp_saver))

    if callbacks:
        all_callbacks += callbacks

    model.fit(train_ds.repeat(), epochs=epochs, verbose=0, validation_data=val_ds, callbacks=all_callbacks,
              steps_per_epoch=epoch_size, class_weight=class_weight)
    if epochs > 0:
        batch_size = len(next(iter(train_ds.take(1).get_single_element()[0].values())))
        save_model(model, model_naming(starting_epoch + epochs), label_enc, batch_size=batch_size)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test evaluation accuracy after {epochs} epochs = {accuracy}")

    # predictions = model.predict(test_ds)
    # predictions = [(np.argmax(x), max(x)) for x in predictions]
    # print(predictions)
    # labels = [l for f, l in test_ds.unbatch().as_numpy_iterator()]
    # lvocab = label_enc.get_vocabulary()
    # data = [[lvocab[x] for x in [l, p]] + [c] for l, (p, c) in zip(labels, predictions)]
    # print(tabulate.tabulate(data, headers='actual predicted certainty'.split()))


def save_model(model: tf.keras.models.Model, path, label_enc: StringLookup, **kwargs):
    model.save(path)
    with open(path / 'label_enc.json', 'w') as f:
        config = label_enc.get_config()
        config['vocabulary'] = label_enc.get_vocabulary()
        json.dump(config, f)
    custom_objects = {'history': model.history.history, 'optimizer': tf.optimizers.serialize(model.optimizer)}
    custom_objects.update(kwargs)

    class Float32Encoder(JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, np.float32):
                return float(o)
            return o

    with open(path / 'custom_objects.json', 'w') as f:
        json.dump(custom_objects, f, cls=Float32Encoder)


def load_model(path: Path):
    model = tf.keras.models.load_model(path)
    encoder = None
    enc_path = path / 'label_enc.json'
    if enc_path.exists():
        with open(enc_path, 'r') as f:
            encoder = StringLookup.from_config(json.load(f))

    custom_objects = {}
    co_file = path / 'custom_objects.json'
    if co_file.exists():
        with open(co_file, 'r') as f:
            custom_objects = json.load(f)
    return model, encoder, custom_objects


class EpochsProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, bar: progressbar.ProgressBar, starting_epoch=0):
        super().__init__()
        self.bar = bar
        self.starting_epoch = starting_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.bar.update(epoch + self.starting_epoch, **{k: str(v)[:6] for k, v in logs.items()})


class SaveEveryNEpochs(tf.keras.callbacks.Callback):

    def __init__(self, period: int, naming: Callable[[Any], pathlib.Path | str], label_enc: StringLookup,
                 starting_epoch=0, remove_prev=False, backup: tf.keras.callbacks.Callback = None):
        super().__init__()
        self.backup = backup
        self.last_saved_path = None
        self.remove_prev = remove_prev
        self.label_enc = label_enc
        self.period = period
        self.naming = naming
        self.starting_epoch = starting_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            epoch += self.starting_epoch
            if epoch % self.period == 0:
                try:
                    model_path = self.naming(epoch)
                    save_model(self.model, model_path, label_enc=self.label_enc)
                    if self.remove_prev and self.last_saved_path:
                        shutil.rmtree(self.last_saved_path)
                    self.last_saved_path = model_path
                except:
                    pass
