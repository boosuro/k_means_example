# import matplotlib,numpy and sklearn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")
#importing model from scikit-learn
from sklearn.cluster import KMeans

#Plotting and visualizing data prior to feeding into machine learning algorithm
x = [1,5,1.5,8,1,9]
y = [2,8,1.8,8,0.6,11]

plt.scatter(x,y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()

# converting data into numpy array
X=np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])

#Initializing K-Means Algorithm with the required parameter
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

#Getting values of centroid and labels based on fitment
centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print(X)
print(centroids)
print(labels)

#Plotting and visualizing data
colors = ['g.','r.','c.','y.']

for i in range(len(X)):
    print("Coordinate : ",X[i],"Label :",labels[i])
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)

plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5,zorder=10)
plt.show()