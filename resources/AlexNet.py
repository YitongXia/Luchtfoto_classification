# import information
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import rasterio


# normalizing and standardizing the images, input should be 227 * 227 image
# so the input data should be square.
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label


def read_single_raster(filename):
    with rasterio.open(filename) as dataset:
        image_band = []
        for i in range(len(dataset.indexes)):
            image_band.append(dataset.read(i + 1))
    return np.array(image_band)


def read_multiple_raster(input_folder):
    # function to read the file and give the info needed
    # input_folder = os.getcwd() + r"..\dataset\segmentation"
    print("activate data folder: ")
    print(input_folder)

    raster_collection = []
    # loop trough files and retrieve objects info as well as loose point info for output
    for file in os.listdir(input_folder):
        file_name = r"..\dataset\segmentation" + "\\" + file
        print("reading file: " + file_name)
        raster_collection.append(read_single_raster(file_name))

    # need to add a reshape function
    

    return np.array(raster_collection)


def get_files(file_dir):
    sun = []
    label_sun = []
    cac = []
    label_cac = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if 'sun' in name[0]:
            sun.append(file_dir + file)
            label_sun.append(0)
        else:
            if 'cac' in name[0]:
                cac.append(file_dir + file)
                label_cac.append(1)
        image_list = np.hstack((sun, cac))
        label_list = np.hstack((label_sun, label_cac))


# image_W ,image_H 指定图片大小，batch_size 每批读取的个数 ，capacity队列中 最多容纳元素的个数
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换数据为 ts 能识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 将image 和 label 放倒队列里
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    # 读取图片的全部信息
    image_contents = tf.read_file(input_queue[0])
    # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
    image = tf.image.decode_png(image_contents, channels=3)
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
    image = tf.image.per_image_standardization(image)

    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=capacity)

    # 重新定义下 label_batch 的形状
    label_batch = tf.reshape(label_batch, [batch_size])
    # 转化图片
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


if __name__ == '__main__':

    # file name (for testing)
    file_name = r"..\dataset\segmentation" + r"\1_black.tif"

    # folder name of the input roof images
    input_folder = r"..\dataset\segmentation1"

    raster_collection = read_multiple_raster(input_folder)
    raster_col = np.ndarray(raster_collection)



    # dataset settings

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # reference the class name of the images during the visualization stage
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 50,000 training data and 10,000 test data by default.
    # the validation is obtained by taking 5000 images within the training data.
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    # define a pipeline to process, using tf.data.Dataset API.
    # tf.data.Dataset.from_tensor_slices methods to train, test,
    # validation data partitions and return a corresponding tensorflow data representation.
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    # prepocessing:
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(train_ds.take(5)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

# for basic data pipeline:
# 1 prepocessing the data within the dataset
# 2 shuffle the dataset
# 3 batch data within the dataset

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))

    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=32, drop_remainder=True))


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
        keras.layers.Dense(10, activation='softmax')
    ])




