# import file
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets

import rasterio
import rasterio.features
import rasterio.warp


# --*** read images ***----
# @ use rasterio to read and store the image
# @ return
# @ numpy array with every band of the image
def read_single_raster(filename):
    with rasterio.open(filename) as dataset:
        image_band = []
        for i in range(len(dataset.indexes)):
            image_band.append(dataset.read(i + 1))
    return image_band


# --*** read images ***----
# use rasterio to read and store the image
# return
# numpy array with every band of the image

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
    return raster_collection


# function to apply k-means clustering on multiple roofs, and draw the plot.
# input: the folder name of the input raster, the number of the clustering.
# output: classification result and plot.

def multiple_raster_kmeans(raster_collection, n_cluster):

    roof_rgb = []
    for raster in raster_collection:
        majority_pixel, second_majority_pixel=single_classification(raster, 6)
        roof_rgb.append(majority_pixel)

    roof_rgb = np.array(roof_rgb)

    X = roof_rgb[:, 0]
    Y = roof_rgb[:, 1]
    Z = roof_rgb[:, 2]

    y_pred = KMeans(n_clusters=n_cluster, random_state=5).fit_predict(roof_rgb)

    input_folder = r"..\dataset\segmentation"
    evaluation(y_pred,input_folder,n_cluster)
    draw_plot(X, Y, Z, y_pred)


# just a test function
# function to apply k-means clustering, and draw the plot.
# input: array of pixel values of roof.
# output: classification result and plot.

def multiple_raster_DBSCAN(raster_collection, esp):

    roof_rgb = []
    for raster in raster_collection:
        majority_pixel, second_majority_pixel=single_classification(raster, 6)
        roof_rgb.append(majority_pixel)

    roof_rgb = np.array(roof_rgb)

    X = roof_rgb[:, 0]
    Y = roof_rgb[:, 1]
    Z = roof_rgb[:, 2]

    y_pred = DBSCAN(eps=esp, min_samples=15).fit_predict(roof_rgb)


# plot the classification result in 3D space based on spectral features
# input: X, Y, Z and y_prediction
# output: plot

def draw_plot(X, Y, Z, y_pred):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=y_pred)

    # set axes limits
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    # set labels
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    plt.show()


# auto classification for each rooftop
# input:tif file of each rooftop, and the number of clustering
# output: (R,G,B) spectral feature of the rooftop

def single_classification_from_file(file_name, n_cluster):
    roof = []
    roof = read_single_raster(file_name)

    # for band in range(len(roof)):
    # clean no_data value (as 256)
    X_old = roof[0].flatten()
    Y_old = roof[1].flatten()
    Z_old = roof[2].flatten()
    roof_rgb = []
    for i in range(len(X_old)):
        if X_old[i] == 256 & Y_old[i] == 256 & Z_old[i] == 256:
            continue
        else:
            roof_rgb.append([X_old[i], Y_old[i], Z_old[i]])

    roof_rgb = np.array(roof_rgb)

    X = roof_rgb[:, 0]
    Y = roof_rgb[:, 1]
    Z = roof_rgb[:, 2]

    y_pred = KMeans(n_clusters=n_cluster, random_state=5).fit_predict(roof_rgb)

    vote = []
    for i in range(n_cluster):
        vote.append(0)

    for item in y_pred:
        vote[item - 1] += 1

    cluster_1 = 0

    for i in range(len(vote)):
        if vote[i] > vote[cluster_1]:
            cluster_1 = i

    cluster_2 = 0
    if cluster_1 == 0:
        cluster_2 = 1
    else:
        cluster_2 = 0

    for i in range(len(vote)):
        if vote[i] > vote[cluster_2]:
            if vote[i] != vote[cluster_1]:
                cluster_2 = i

    majority_pixel = [0, 0, 0]
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == cluster_1:
            majority_pixel[0] += X[i]
            majority_pixel[1] += Y[i]
            majority_pixel[2] += Z[i]
            count += 1
    majority_pixel = [majority_pixel[0] / count,
                      majority_pixel[1] / count,
                      majority_pixel[2] / count]

    second_majority_pixel = [0, 0, 0]
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == cluster_2:
            second_majority_pixel[0] += X[i]
            second_majority_pixel[1] += Y[i]
            second_majority_pixel[2] += Z[i]
            count += 1
    second_majority_pixel = [second_majority_pixel[0] / count,
                             second_majority_pixel[1] / count,
                             second_majority_pixel[2] / count]

    return majority_pixel, second_majority_pixel

def single_classification(roof, n_cluster):

    # for band in range(len(roof)):
    # clean no_data value (as 256)
    X_old = roof[0].flatten()
    Y_old = roof[1].flatten()
    Z_old = roof[2].flatten()
    roof_rgb = []
    for i in range(len(X_old)):
        if X_old[i] == 256 & Y_old[i] == 256 & Z_old[i] == 256:
            continue
        else:
            roof_rgb.append([X_old[i], Y_old[i], Z_old[i]])

    roof_rgb = np.array(roof_rgb)

    X = roof_rgb[:, 0]
    Y = roof_rgb[:, 1]
    Z = roof_rgb[:, 2]

    y_pred = KMeans(n_clusters=n_cluster, random_state=5).fit_predict(roof_rgb)

    vote = []
    for i in range(n_cluster):
        vote.append(0)

    for item in y_pred:
        vote[item - 1] += 1

    cluster_1 = 0

    for i in range(len(vote)):
        if vote[i] > vote[cluster_1]:
            cluster_1 = i

    cluster_2 = 0
    if cluster_1 == 0:
        cluster_2 = 1
    else:
        cluster_2 = 0

    for i in range(len(vote)):
        if vote[i] > vote[cluster_2]:
            if vote[i] != vote[cluster_1]:
                cluster_2 = i

    majority_pixel = [0, 0, 0]
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == cluster_1:
            majority_pixel[0] += X[i]
            majority_pixel[1] += Y[i]
            majority_pixel[2] += Z[i]
            count += 1
    majority_pixel = [majority_pixel[0] / count,
                      majority_pixel[1] / count,
                      majority_pixel[2] / count]

    second_majority_pixel = [0, 0, 0]
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == cluster_2:
            second_majority_pixel[0] += X[i]
            second_majority_pixel[1] += Y[i]
            second_majority_pixel[2] += Z[i]
            count += 1
    second_majority_pixel = [second_majority_pixel[0] / count,
                             second_majority_pixel[1] / count,
                             second_majority_pixel[2] / count]

    return majority_pixel, second_majority_pixel


# function to read the file and give the info needed
# return:
# the array of the RGB value of all the roofs
def read(file_route):
    input_file = os.getcwd() + file_route
    print("activate data folder: ")
    print(input_file)
    roof_rgb = []
    roofs_rgb = []
    roofs_gid = []

    with open(input_file, 'r') as f:
        for line in f.readlines():
            split = line.split()
            roofs_gid.append(split[0])
            roof_rgb = [float(split[1]), float(split[2]), float(split[3])]
            roofs_rgb.append(roof_rgb)
    return roofs_gid, roofs_rgb


# function to apply k-means clustering, and draw the plot.
# input: array of pixel values of roof.
# output: classification result and plot.
def kmeans(roofs_rgb,n_cluster):
    roofs_rgb = np.array(roofs_rgb)
    X = roofs_rgb[:, 0]
    Y = roofs_rgb[:, 1]
    Z = roofs_rgb[:, 2]
    y_pred = KMeans(n_clusters=n_cluster, random_state=5).fit_predict(roofs_rgb)
    draw_plot(X, Y, Z, y_pred)


# function to apply DBSCAN clustering, and draw the plot.
# input: array of pixel values of roof.
# output: classification result and plot.
def dbscan(roofs_rgb):
    roofs_rgb = np.array(roofs_rgb)
    X = roofs_rgb[:, 0]
    Y = roofs_rgb[:, 1]
    Z = roofs_rgb[:, 2]
    y_pred = DBSCAN(eps=3, min_samples=15).fit_predict(roofs_rgb)
    draw_plot(X, Y, Z, y_pred)


# this function is to evaluate the clustering result
# compare with the ground truth label
def evaluation(y_pred, input_folder,n_cluster):
    file_name = []
    for file in os.listdir(input_folder):
        file_name.append(file)
    # for i in range(len(y_pred)):
    #     print("this is {}, the file name is {}".format(y_pred[i], file_name[i]))

    for i in range(n_cluster):
        for j in range(len(y_pred)):
            if i == y_pred[j]:
                print("the clustering result is {}, and the file name is {}".format(y_pred[j], file_name[j]))


if __name__ == '__main__':
    # file name (for testing)
    file_name = r"..\dataset\segmentation" + r"\2_solar.tif"

    # folder name of the input roof images
    input_folder = r"..\dataset\segmentation"

    raster_collection = read_multiple_raster(input_folder)
    multiple_raster_kmeans(raster_collection, 7)
    print("the clustering process is finished.")

    # major, second_major = single_classification_from_file(file_name, 6)
    # print("the first color is: ", major)
    # print("the second color is: ", second_major)
