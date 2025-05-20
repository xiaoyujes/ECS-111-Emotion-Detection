import tensorflow as tf
import pandas as pd
import os

# Base paths (relative)
base_dir = 'ECS-111-Emotion-Detection/data'
training_csv = os.path.join(base_dir, 'training.csv')
testing_csv = os.path.join(base_dir, 'testing.csv')
converted_dir = os.path.join(base_dir, 'converted')
img_size = (224, 224)

# Load CSVs
train_df = pd.read_csv(training_csv)
test_df = pd.read_csv(testing_csv)

# Resolve full paths
train_df['imagepath'] = train_df['imagepath'].apply(lambda x: os.path.join(converted_dir, x))
test_df['imagepath'] = test_df['imagepath'].apply(lambda x: os.path.join(converted_dir, x))

# Label encoding
label_names = sorted(train_df['label'].unique())
label_to_index = {name: i for i, name in enumerate(label_names)}
train_df['label'] = train_df['label'].map(label_to_index)
test_df['label'] = test_df['label'].map(label_to_index)

# Preprocess function
def preprocess(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image, label

# Dataset builder
def build_dataset(df):
    paths = df['imagepath'].tolist()
    labels = df['label'].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda path, label: preprocess(path, label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return ds

# Final datasets
train_ds = build_dataset(train_df)
test_ds = build_dataset(test_df)