import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

dataset = tfds.load('wikipedia/20230601.id', split="train")

def preprocess(sample):
    text = sample['text'].numpy().decode('utf-8')
    title = sample['title'].numpy().decode('utf-8')
    return text, title

def dataset_generator(dataset):
    for sample in dataset:
        yield preprocess(sample)

train_data = tf.data.Dataset.from_generator(
    lambda: dataset_generator(dataset), output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.string))
)

vectorizer = TextVectorization(max_tokens=20000, output_mode='int', output_sequence_length=100)
text_ds = train_data.map(lambda x, y: x)
vectorizer.adapt(text_ds.batch(128))

def vectorize_text(text, label):
    text = vectorizer(text)
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform([label.numpy().decode('utf-8')])
    return text, label

train_data = train_data.map(lambda x, y: tf.py_function(func=vectorize_text, inp=[x, y], Tout=(tf.int64, tf.int64)), num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.map(lambda x, y: (tf.ensure_shape(x, [100]), tf.ensure_shape(y, [1])))
train_data = train_data.shuffle(10000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    layers.Input(shape=(100,), dtype="int64"),
    layers.Embedding(input_dim=20000, output_dim=128),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_data, epochs=120)
model.save('model.h5')
