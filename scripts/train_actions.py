import tensorflow as tf

from ml.ml_util import create_input_pipeline, train_model, create_data_action
from util.core import find_root_dir

model_type = 'action'
model_ver = 'v4'
model_epochs = 5000
model_name_suffix = '_100x5'
epochs = 10000

tf.get_logger().setLevel('INFO')
train_ds, val_ds, test_ds, fields_map, label_enc = create_data_action(model_ver)
[(train_features, label_batch)] = train_ds.take(1)

all_inputs = []
encoded_features = []
for feature in train_features.keys():
    field = fields_map[feature]
    inp, encoded = create_input_pipeline(field, train_ds)
    all_inputs.append(inp)
    encoded_features.append(encoded)
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(30, activation="relu")(all_features)
x = tf.keras.layers.Dense(30, activation="relu")(x)
# x = tf.keras.layers.Dense(30, activation="relu")(x)
# x = tf.keras.layers.Dense(50, activation="relu")(x)
# x = tf.keras.layers.Dense(50, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(label_enc.vocabulary_size(), activation='softmax')(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc'])
model_prefix = f"{model_type}_{model_ver}{model_name_suffix}_"
img_dir = find_root_dir() / 'model/_img'
img_dir.mkdir(parents=True, exist_ok=True)
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", show_dtype=True,
                          to_file=img_dir / f"{model_prefix}.png")
train_model(model, train_ds, val_ds, test_ds, epochs, label_enc, model_prefix, save_each_n_epochs=1000,
            checkpoint_every_n_epochs=100)
