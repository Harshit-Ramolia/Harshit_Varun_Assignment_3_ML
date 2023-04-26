import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

st.title("Hierarchical Clustering")

st.write("Create a slider for each step in hierarchical clustering of unsupervised data. Use the sklearn.datasets make_blobs function to create the dataset and visualize the predictions using a changing voronoi diagram.")

number = st.sidebar.number_input("Number of points", min_value=0, max_value=50, value=10)
features, _clusters = make_blobs(n_samples = number, n_features = 2, centers = 3, cluster_std = 1.3, shuffle = True)


step = st.sidebar.slider("Hierarchical step", min_value=3, max_value=number)

st.sidebar.write("Created by : Harshit Ramolia and Varun Barala")

hierarchical_cluster = AgglomerativeClustering(n_clusters=step , affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(features)

clf = NearestCentroid()
clf.fit(features, labels)

centers = clf.centroids_
vor = Voronoi(centers)
# plt.figure()
# ax = plt.Subplot(fig, 111)
fig, ax = plt.subplots()

voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=1, point_size=2)
ax.axis([min(features[:, 0])*1.2, max(features[:, 0])*1.2, min(features[:, 1])*1.2, max(features[:, 1])*1.2 ])

# ax.scatter([1, 2, 3], [1, 2, 3])
   
ax.scatter(features[:, 0], features[:, 1], c=labels)
# plt.box(False)
# plt.show()
st.pyplot(fig)

st.write("Hierarchical clustering is a process of grouping similar data points together based on their distance from each other. The slider allows the user to change the number of clusters and see how the data is grouped at each step of the clustering process.")
st.write("As the user changes the slider, the number of clusters changes, and the voronoi diagram updates to show how the data is grouped based on the new number of clusters. The voronoi diagram is a visual representation of the clusters, where each point in the diagram represents a data point, and the color of the point represents which cluster it belongs to. As the slider is moved, the colors of the points change to reflect the new cluster assignments.")
st.write("This tool can be useful for exploring unsupervised data and understanding how different clustering algorithms group data points together. It allows users to visualize the clustering process step by step and gain insights into how the algorithm is grouping the data.")