from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn
import matplotlib.pyplot as plt
# 生成随机模拟二维数据
x,y = make_blobs(random_state=1)
# 构建聚类模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
# print("Cluster memberships:\n{}".format(kmeans.labels_))
mglearn.discrete_scatter(x[:,0],x[:,1],kmeans.labels_,markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],[0,1,2],markers="^",markeredgewidth=2)
plt.show()