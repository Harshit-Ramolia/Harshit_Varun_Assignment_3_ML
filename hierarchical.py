import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(1234)

st.title("Harshit")

number = st.sidebar.number_input("Number of points", min_value=0, max_value=50, value=10)
features, _clusters = make_blobs(n_samples = number, n_features = 2, centers = 3, cluster_std = 1.3, shuffle = True)


step = st.sidebar.slider("Hierarchical step", min_value=3, max_value=number)

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