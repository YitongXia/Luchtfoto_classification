
#import file
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets


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
    all_X= []
    all_Y=[]
    all_Z=[]

    with open(input_file, 'r') as f:

        for line in f.readlines():

            split = line.split()
            roofs_gid.append(split[0])
            roof_rgb = [float(split[1]), float(split[2]), float(split[3])]
            all_X.append(float(split[1]))
            all_Y.append(float(split[2]))
            all_Z.append(float(split[3]))
            roofs_rgb.append(roof_rgb)
    return roofs_gid,roofs_rgb,all_X,all_Y,all_Z

def kmeans(X,Y,Z,roofs_rgb):
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

def dbscan(X,Y,Z,roofs_rgb):
    y_pred = DBSCAN(eps=3,min_samples=15).fit_predict(roofs_rgb)

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

    file_route= r"\Dataset" + r"\rgb.txt"
    roofs_rgb=[]
    roofs_gid=[]
    X=[]
    Y=[]
    Z=[]
    roofs_gid, roofs_rgb,X,Y,Z=read(file_route)
    dbscan(X,Y,Z,roofs_rgb)



