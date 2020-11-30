#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:59:23 2020

@author: tan
"""

import matplotlib.pyplot as plt
import numpy as np
import random

def cal_dis(dataset,center):
    center_mat = np.tile(center,(dataset.shape[0],1))
    distance_mat = np.square(dataset[:,0] - center_mat[:,0])+np.square(dataset[:,1] - center_mat[:,1])+np.square(dataset[:,2] - center_mat[:,2])
    return distance_mat
                        

def kmeans(dataset, num_clusters, max_iter):
    #Set initial center
    num_points = dataset.shape[0]
    random.seed(1)
    centers_idx = random.sample(range(0,num_points-1),num_clusters)
    iteration = 0
    while(1):
        dis_mat = []
        for i in range(num_clusters):
            center_tmp = np.expand_dims(dataset[centers_idx[i]],0) #dataset[centers_idx[i]]->(3,)
            dis = cal_dis(dataset,center_tmp)
            dis_mat.append(dis)
        dis_mat = np.array(dis_mat)
        min_dis = np.argmin(dis_mat,axis=0)
        if iteration == max_iter:
            return min_dis
        new_center = []
        for c in range(num_clusters):
            c_index = np.where(min_dis==c)
            cur_data = dataset[c_index]
            mean_data = np.mean(cur_data,axis=0)
            new_center.append(np.argmin(cal_dis(dataset,np.expand_dims(mean_data,0))))
        centers_idx = new_center
        iteration += 1
            
            
        
        
        
        
        
        

#generate a scatter plot for each cluster individually
def scatter_cluster(points, ax):
    X = []
    Y = []
    Z = []
    for point in points:
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])

    #Generate a random color to assign to cluster in question
    r = lambda: random.randint(0,255)
    ax.scatter(X,Y,Z, c = '#%02X%02X%02X' % (r(),r(),r()))

#Show the final scatterplot with the different points colored according to cluster
def show_plot(points,res,num_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_clusters):
        idx = np.where(res == i)
        scatter_cluster(points[idx], ax)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()


data = np.load('visulization/ori_pc.npy')
num_clusters = 10
res = kmeans(data,num_clusters,2)
show_plot(data,res,num_clusters)
res = kmeans(data,num_clusters,50)
show_plot(data,res,num_clusters)
