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
        print(dataset.indexes)
        for i in range(len(dataset.indexes)):
            image_band.append(dataset.read(i + 1))
    return image_band


# --*** read images ***----
# @ use rasterio to read and store the image
# @ return
# @ numpy array with every band of the image

def read_raster_folder(input_folder):
    # function to read the file and give the info needed
    # input_folder = os.getcwd() + r"..\dataset\segmentation"
    print("activate data folder: ")
    print(input_folder)

    raster_collection = []
    # loop trough files and retrieve objects info as well as loose point info for output
    for file in os.listdir(input_folder):
        raster_collection.append(read_single_raster(file))
    return raster_collection


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
# input:tif file of each rooftop
# output: (R,G,B) spectral feature of the rooftop

def single_classification(file_name, n_cluster):
    roof = []
    roof = read_single_raster(file_name)

    # for band in range(len(roof)):
    X_old = roof[0].flatten()
    Y_old = roof[1].flatten()
    Z_old = roof[2].flatten()
    roof_rgb = []
    X = []
    Y = []
    Z = []
    for i in range(len(X_old)):
        if X_old[i] == 256 & Y_old[i] == 256 & Z_old[i] == 256:
            continue
        else:
            roof_rgb.append([X_old[i], Y_old[i], Z_old[i]])
            X.append(X_old[i])
            Y.append(Y_old[i])
            Z.append(Z_old[i])

    print("the number of valid pixel is ", len(X))
    # clean no_data value (as 256)
    y_pred = KMeans(n_clusters=n_cluster, random_state=5).fit_predict(roof_rgb)
    # draw_plot(X,Y,Z,y_pred)

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

    print("the main cluster is: ", cluster_1)
    print("the second cluster is: ", cluster_2)

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
    all_X = []
    all_Y = []
    all_Z = []

    with open(input_file, 'r') as f:
        for line in f.readlines():
            split = line.split()
            roofs_gid.append(split[0])
            roof_rgb = [float(split[1]), float(split[2]), float(split[3])]
            all_X.append(float(split[1]))
            all_Y.append(float(split[2]))
            all_Z.append(float(split[3]))
            roofs_rgb.append(roof_rgb)
    return roofs_gid, roofs_rgb, all_X, all_Y, all_Z


def kmeans(X, Y, Z, roofs_rgb):
    y_pred = KMeans(n_clusters=3, random_state=5).fit_predict(roofs_rgb)
    draw_plot(X, Y, Z, y_pred)


def dbscan(X, Y, Z, roofs_rgb):
    y_pred = DBSCAN(eps=3, min_samples=15).fit_predict(roofs_rgb)
    draw_plot(X, Y, Z, y_pred)


if __name__ == '__main__':
    file_name = r"..\dataset\segmentation" + r"\15_tile.tif"

    input_folder = os.getcwd() + r"..\dataset\segmentation"

    major, second_major= single_classification(file_name, 6)
    print("the first color is: ",major)
    print("the second color is: ",second_major)