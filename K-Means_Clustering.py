# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:02:48 2023

@author: nikhilve

Here the customer has asked us to build a clustering algo only based upon Annual Income and spending score.
The mall owner has no idea on how many groups to be formed.


It is Unsupervised Learning

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Reading the CSV data
data = pd.read_csv('Mall_Customers.csv')


#Here x is the matrix of column. Science it is Unsupervised learning
x=data.iloc[:,[3,4]].values


#Elbow method using WCSS.
from sklearn.cluster import KMeans

#Creating a empty list to hold the values of wcss.
wcss=[]
#WCSS= within cluster sum of squares.
#k-means++ helps to overcome random allocation trap.
#n_init means how many times we want to reinitialize the centroids from the starting
#n_clusters is the no of clusters we want to begin with initially.
#max_iter is the maximum no of times the centroids are getting reassigned when the data points are changing groups.
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10 )
    #final formation of the clusters is taken as the iteration where there was least movement of data points in any initialization(n_iter).
    #i.e. the cluster at max_iter=300(will be taken) from each initialization n_init and then compared, cluster with the least movent of centroids will be selected as final cluster.
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#Here the inertia method is used to give the wcss value and append it in wcss[](empty list).


#Plotting the graph of no of clusters vs WCSS-values to get the elbow angle.

plt.figure(figsize=(8,5))
plt.subplot(1,2,1)#This plots the graph of wcss vs no of clusters. Here 1=no of rows, 2= no of columns and 1= 1st plot i.e. elbow graph

plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("wcss-values")
plt.title("Graph of WCSS vs no of clusters")

#Implementing K-Means algorithm with K=5, using this value from elbow graph.
kmeans= KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10 )

#Fitting and predicting the clusters for each customer
y_means = kmeans.fit_predict(x)

#Visual representation of the cluster-formation
plt.subplot(1,2,2)#to print the scatter plot of salary vs income.Here 1=no of rows, 2= no of columns and 2= 2nd plot i.e. income vs spending

plt.scatter(x[y_means==0,0], x[y_means==0,1], s=50, c='red', label="cluster1")
plt.scatter(x[y_means==1,0], x[y_means==1,1], s=50, c='blue', label="cluster2")
plt.scatter(x[y_means==2,0], x[y_means==2,1], s=50, c='green', label="cluster3")
plt.scatter(x[y_means==3,0], x[y_means==3,1], s=50, c='yellow', label="cluster4")
plt.scatter(x[y_means==4,0], x[y_means==4,1], s=50, c='pink', label="cluster5")
#We can rename the labels as per our requirment.

#Plotting Centroids by using the average values of x-coordinate(here annual salary) and y-coordinate(Here Purchase score)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c="cyan", label="Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Clusters of Customers")
plt.legend()
plt.tight_layout()
plt.show()





















