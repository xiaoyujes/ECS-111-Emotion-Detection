#!/usr/bin/env python3
"""
localized_model_algorithm.py

Localized Model Algorithm File

This script implements the core model architecture, 
training loop (with data augmentation), and best‐model 
checkpoint saving, all adapted for the local directory 
structure and dataset paths.
"""
'''
Much of the algorithm code is inspired from this source:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/The-Basics-of-ResNet50---Vmlldzo2NDkwNDE2
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import seaborn as sns

####Hyperparameters###
learningrate = 0.00001 
batchsize = 32
epochs = 100
dropoutrate = 0.35
freezelayer = 70
######################

################
converted_dir = 'C:/Users/ASUS/Desktop/ecs111/final project/converted'
img_size = (224, 224)

df = pd.read_csv('C:/Users/ASUS/Desktop/ecs111/final project/training_onehot.csv')

df['imagepath'] = df['imagepath'].apply(
    lambda x: os.path.normpath(os.path.join(converted_dir, x)).replace('\\', '/') #normalized slashes
)

image_paths = df['imagepath'].tolist()
onehot_labels = df.iloc[:, -4:].values.astype('float32')

def preprocess(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image, label

def build_onehot_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batchsize).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = build_onehot_dataset(image_paths, onehot_labels)
################

'''
Splitting training and validation set for training the model
Training-validation split: 80-20
'''
train_path, val_path, train_label, val_label = train_test_split(image_paths,
                                                                onehot_labels,
                                                                test_size=0.2,
                                                                stratify=df['label_encoded'],
                                                                random_state=42)

train_df = build_onehot_dataset(train_path, train_label)
val_df = build_onehot_dataset(val_path, val_label, shuffle=False)

################

'''
Building the MobileNetV2 Model
'''
PROJECT_ROOT = r"C:\Users\ASUS\Desktop\ecs111\final project"
inputs =  tf.keras.Input(shape=(224, 224, 3)) #tensor output
base_model = MobileNetV2(include_top=False, #weights pretrained on imagenet dataset
                         weights='imagenet', #omits fully connected classification layers
                         input_shape=(224, 224, 3)) #defines shape and color of images

base_model.trainable = True #freezes base model rate so that weights are not updated during training

for layer in base_model.layers[:freezelayer]: #freezes the first 70 layers (freezes weights in first 70 layers)
    layer.trainable = False

data_augmentation = tf.keras.Sequential([ #applies transformation to data for better generalization
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.1, 0.1),],
    name='augmentation'
)

x = data_augmentation(inputs)
x = base_model(x, training=True) #passing input through frozen ResNet50
x = layers.GlobalAveragePooling2D()(x) #averages feature map
x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
#fully connected layers to learn emotion
x = layers.Dropout(dropoutrate)(x) #include dropout
x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = layers.Dropout(dropoutrate)(x)
outputs = layers.Dense(4, activation='softmax')(x) #probabilities across the 4 emotions (4 = number of classes)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)]

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(PROJECT_ROOT, "test", "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
callbacks.append(checkpoint_cb)
labels_encoded = df['label_encoded'].values

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_encoded), y=labels_encoded)
class_weights = dict(enumerate(class_weights))

hist = model.fit(train_df, validation_data=val_df, epochs=epochs, class_weight=class_weights, callbacks=callbacks)

###Evaluation (F1-Score & Confusion Matrix)###
val_labs_true = np.argmax(val_label, axis=1) #converts one-hot index to labels

pred_prob = model.predict(val_df)
pred_lab = np.argmax(pred_prob, axis=1)

f1 = f1_score(val_labs_true, pred_lab, average='macro') #calculates f1 score
print(f'F1-Score: {f1:.4f}')

print(f'Classification Report: \n{classification_report(val_labs_true, pred_lab)}') #classification report

#creating confusion matrix
confusion_mat = confusion_matrix(val_labs_true, pred_lab)
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Angry', 'Fear', 'Happy', 'Sad'],
            yticklabels=['Angry', 'Fear', 'Happy', 'Sad'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('C:/Users/ASUS/Desktop/ecs111/final project/test/confusion_matrix.png')

#plotting accuracy
plt.figure()
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('C:/Users/ASUS/Desktop/ecs111/final project/test/accuracyplot.png')

#plotting loss
plt.figure()
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('C:/Users/ASUS/Desktop/ecs111/final project/test/lossplot.png')