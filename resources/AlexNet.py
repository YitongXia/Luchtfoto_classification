# import information
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import rasterio
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)



def tf_get_files(file_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        file_dir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        image_size=(227, 227),
        shuffle=True,
        seed=None,
        crop_to_aspect_ratio=True,
        batch_size=2
    )
    return ds


if __name__ == '__main__':

    train_file_dir = r"..\dataset\DL\train"
    test_file_dir = r"..\dataset\DL\test"
    validation_file_dir = r"..\dataset\DL\validation"
    train_ds = tf_get_files(train_file_dir)
    test_ds = tf_get_files(test_file_dir)
    validation_ds = tf_get_files(validation_file_dir)


    # (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    # validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    # train_images, train_labels = train_images[5000:], train_labels[5000:]
    # train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    # train_ds = (train_ds
    #             .map(process_images)
    #             .shuffle(buffer_size=train_ds_size)
    #             .batch(batch_size=4, drop_remainder=True))
    #
    # test_ds = (test_ds
    #            .map(process_images)
    #            .shuffle(buffer_size=train_ds_size)
    #            .batch(batch_size=4, drop_remainder=True))
    # validation_ds = (validation_ds
    #                  .map(process_images)
    #                  .shuffle(buffer_size=train_ds_size)
    #                  .batch(batch_size=4, drop_remainder=True))

    # model implementation
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=(227, 227, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])

    root_logdir = os.path.join(os.curdir, "logs\\fit\\")
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    model.summary()

    model.fit(train_ds,
              epochs=10,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=[tensorboard_cb])

    model.evaluate(test_ds)
    model.evaluate(validation_ds)
