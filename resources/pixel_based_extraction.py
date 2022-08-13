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
def read_single_raster(file_name):
    with rasterio.open(file_name) as dataset:
        image_band = []
        for i in range(len(dataset.indexes)):
            image_band.append(dataset.read(1))
    return image_band


def read_raster_folder(input_folder):
    # function to read the file and give the info needed
    # input_folder = os.getcwd() + r"..\dataset\segmentation"
    print("activate data folder: ")
    print(input_folder)

    raster_collection = []
    # loop trough files and retreive objects info as well as loose point info for output
    for file in os.listdir(input_folder):
        raster_collection.append(read_single_raster(file))
    return raster_collection



# @ auto classification for each rooftop
# input:tif file of each rooftop
# output: (R,G,B) spectral feature of the rooftop

def single_classification(file_name):

    roof = []
    roof = read_single_raster(file_name)
    # for band in range(len(roof)):
    X = roof[0].flatten()
    Y = roof[1].flatten()
    Z = roof[2].flatten()
    roof_rgb = []
    for i in range(len(roof[0])):
        roof_rgb.append([X[i], Y[i], Z[i]])

    y_pred = KMeans(n_clusters=9, random_state=5).fit_predict(roof_rgb)

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
    y_pred = KMeans(n_clusters=9, random_state=5).fit_predict(roofs_rgb)

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


def dbscan(X, Y, Z, roofs_rgb):
    y_pred = DBSCAN(eps=3, min_samples=15).fit_predict(roofs_rgb)

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


if __name__ == '__main__':
    file_name = r"..\dataset" + r"\amsterdam.tif"

    input_folder = os.getcwd() + r"..\dataset\segmentation"

    read_raster(file_name)
