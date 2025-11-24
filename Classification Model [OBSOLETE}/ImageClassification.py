import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

(train_data, val_data, test_data), data_info = tfds.load("cifar10", 
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            as_supervised=True, with_info=True, shuffle_files=True)
fig = tfds.show_examples(train_data, data_info)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

y_train = []
for img, label in train_data:
    y_train.append(int(label))

sns.countplot(x=y_train)
plt.show()

data_augmentation = keras.Sequential([
    layers.Resizing(224, 224),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.Rescaling(1.0 / 255.0) 
])
def augment(image, label):
    return data_augmentation(image), label
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
train_data = train_data.map(augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Use a pretrained base model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
# Add custom classifier on top
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Complile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

epochs = 10

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=callbacks
)

# save the model
model.save('cifar10_model.keras')

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Load the model
model = keras.models.load_model('cifar10_model.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

def plot_predictions(images, true_labels, pred_labels, class_names, n=8):
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, 4, i + 1)
        img = images[i].numpy() * 255.0
        img = np.clip(img, 0, 255).astype("uint8")
        plt.imshow(img)
        true_cls = class_names[true_labels[i]]
        pred_cls = class_names[pred_labels[i]]
        plt.title(f"True: {true_cls}\nPred: {pred_cls}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
# Get one batch from test set
sample_images, sample_labels = next(iter(test_data))
sample_preds = model.predict(sample_images, verbose=0)
sample_pred_labels = np.argmax(sample_preds, axis=1)
plot_predictions(sample_images, sample_labels, sample_pred_labels, classes)

# Unfreezing the last 50 layers of the models
base_model.trainable = True
fine_tune = 50
# Freeze all the layers before the `fine_tune` layer
for layer in base_model.layers[:-fine_tune]:
    layer.trainable = False

base_model.summary()

print(f"Trainable Layers: {len(model.trainable_variables)}")

# Training the Model
epochs = 20
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=callbacks
)

# Save the model
model.save('cifar10_model_final_improved.keras')

# Load saved model
model = keras.models.load_model('cifar10_model_final_improved.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

# Accuracy and Predictions
y_true = []
y_pred = []
for images, labels in test_data:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Precision: {precision_score(y_true, y_pred, average='macro')}")
print(f"Recall: {recall_score(y_true, y_pred, average='macro')}")
print(f"Full classification report: {classification_report(y_true, y_pred)}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()