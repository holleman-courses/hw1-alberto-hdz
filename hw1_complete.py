#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


##

def build_model1():
  """Fully-connected model: Flatten + 3 Dense(128, leaky_relu) + Dense(10)"""
  model = Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10)
  ])
  model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

def build_model2():
  """CNN with 6 Conv2D+BatchNorm pairs, Flatten, Dense(10)"""
  model = Sequential([
    # Pair 1: 32 filters, stride=2
    layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu',
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    # Pair 2: 64 filters, stride=2
    layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 3: 128 filters, stride=1 (default)
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 4: 128 filters, stride=1
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 5: 128 filters, stride=1
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 6: 128 filters, stride=1
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Head
    layers.Flatten(),
    layers.Dense(10)
  ])
  model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

def build_model3():
  """Same structure as model2 but ALL conv layers are SeparableConv2D"""
  model = Sequential([
    # Pair 1: 32 filters, stride=2
    layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu',
                           input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    # Pair 2: 64 filters, stride=2
    layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 3: 128 filters, stride=1
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 4: 128 filters, stride=1
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 5: 128 filters, stride=1
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Pair 6: 128 filters, stride=1
    layers.SeparableConv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Head
    layers.Flatten(),
    layers.Dense(10)
  ])
  model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

def build_model50k():
  """Best model with <=50,000 parameters targeting >=60% accuracy"""
  model = Sequential([
    # Use a small regular Conv2D first
    layers.Conv2D(16, (3,3), padding='same', activation='relu',
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    # Separable convolutions to keep param count low
    layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(48, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(80, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(96, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Global average pooling instead of flatten (saves tons of params)
    layers.GlobalAveragePooling2D(),
    layers.Dense(10)
  ])
  model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Load the CIFAR10 data set
  (full_train_images, full_train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

  # Squeeze labels from shape (N, 1) to (N,)
  full_train_labels = full_train_labels.squeeze()
  test_labels = test_labels.squeeze()

  # Normalize pixel values to [0, 1]
  full_train_images = full_train_images.astype('float32') / 255.0
  test_images = test_images.astype('float32') / 255.0

  # Split training into train (45000) + validation (5000)
  train_images = full_train_images[:45000]
  train_labels = full_train_labels[:45000]
  val_images = full_train_images[45000:]
  val_labels = full_train_labels[45000:]

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

  ########################################
  ## Build and train model 1
  print("\n" + "="*60)
  print("MODEL 1: Fully Connected")
  print("="*60)
  model1 = build_model1()
  model1.summary()
  history1 = model1.fit(train_images, train_labels, epochs=30,
                        validation_data=(val_images, val_labels))
  test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
  print(f"\nModel 1 Results:")
  print(f"  Training accuracy:   {history1.history['accuracy'][-1]:.4f}")
  print(f"  Validation accuracy: {history1.history['val_accuracy'][-1]:.4f}")
  print(f"  Test accuracy:       {test_acc1:.4f}")

  ########################################
  ## Build and train model 2
  print("\n" + "="*60)
  print("MODEL 2: CNN with Conv2D")
  print("="*60)
  model2 = build_model2()
  model2.summary()
  history2 = model2.fit(train_images, train_labels, epochs=30,
                        validation_data=(val_images, val_labels))
  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
  print(f"\nModel 2 Results:")
  print(f"  Training accuracy:   {history2.history['accuracy'][-1]:.4f}")
  print(f"  Validation accuracy: {history2.history['val_accuracy'][-1]:.4f}")
  print(f"  Test accuracy:       {test_acc2:.4f}")

  ########################################
  ## Test image classification
  # Uncomment and update the filename below after you add your test image
  # test_img = np.array(keras.utils.load_img(
  #     './test_image_cat.jpg',
  #     grayscale=False, color_mode='rgb', target_size=(32, 32)))
  # test_img_normalized = test_img.astype('float32') / 255.0
  # prediction = model2.predict(test_img_normalized[np.newaxis, ...])
  # predicted_class = class_names[np.argmax(prediction)]
  # print(f"\nTest image predicted class: {predicted_class}")

  ########################################
  ## Build and train model 3
  print("\n" + "="*60)
  print("MODEL 3: CNN with SeparableConv2D")
  print("="*60)
  model3 = build_model3()
  model3.summary()
  history3 = model3.fit(train_images, train_labels, epochs=30,
                        validation_data=(val_images, val_labels))
  test_loss3, test_acc3 = model3.evaluate(test_images, test_labels)
  print(f"\nModel 3 Results:")
  print(f"  Training accuracy:   {history3.history['accuracy'][-1]:.4f}")
  print(f"  Validation accuracy: {history3.history['val_accuracy'][-1]:.4f}")
  print(f"  Test accuracy:       {test_acc3:.4f}")

  ########################################
  ## Build and train best model (<=50k params)
  print("\n" + "="*60)
  print("BEST MODEL: <=50k parameters")
  print("="*60)
  model50k = build_model50k()
  model50k.summary()

  # Data augmentation for better accuracy
  data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
  ])

  # Train for more epochs since it's a smaller model
  # Use augmented data
  history50k = model50k.fit(
    data_augmentation(train_images), train_labels,
    epochs=60,
    validation_data=(val_images, val_labels)
  )
  test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels)
  print(f"\nBest Model Results:")
  print(f"  Training accuracy:   {history50k.history['accuracy'][-1]:.4f}")
  print(f"  Validation accuracy: {history50k.history['val_accuracy'][-1]:.4f}")
  print(f"  Test accuracy:       {test_acc50k:.4f}")
  print(f"  Total parameters:    {model50k.count_params()}")

  # Save the best model
  model50k.save("best_model.h5")
  print("Best model saved to best_model.h5")

  ########################################
  ## Print comparison table
  print("\n" + "="*60)
  print("MODEL COMPARISON TABLE")
  print("="*60)
  print(f"{'Model':<12} {'Params':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
  print("-" * 60)
  print(f"{'Model 1':<12} {model1.count_params():<12} {history1.history['accuracy'][-1]:<12.4f} {history1.history['val_accuracy'][-1]:<12.4f} {test_acc1:<12.4f}")
  print(f"{'Model 2':<12} {model2.count_params():<12} {history2.history['accuracy'][-1]:<12.4f} {history2.history['val_accuracy'][-1]:<12.4f} {test_acc2:<12.4f}")
  print(f"{'Model 3':<12} {model3.count_params():<12} {history3.history['accuracy'][-1]:<12.4f} {history3.history['val_accuracy'][-1]:<12.4f} {test_acc3:<12.4f}")
  print(f"{'Best(<50k)':<12} {model50k.count_params():<12} {history50k.history['accuracy'][-1]:<12.4f} {history50k.history['val_accuracy'][-1]:<12.4f} {test_acc50k:<12.4f}")
