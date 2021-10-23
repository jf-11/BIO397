
# K-MEANS
############################################################################

using ML
using ScikitLearn
using Plots
@sk_import datasets: make_blobs
features, labels = make_blobs(n_samples=500, centers=3, cluster_std=0.55, random_state=0)

############################################################################

# look at the data first
x = features[:, 1]
y = features[:, 2]
theme(:ggplot2;)
scatter(x,y,title="x vs. y",label=false)
xlabel!("x")
ylabel!("y")

############################################################################

# now run or k-means algorithm
k, cluster = ML.k_means_clustering(features,3)

# extract clusters
x_clust = k[:,1]
y_clust = k[:,2]

# extract labels
labels = cluster[:,3]
labels = string.(labels)

# plot again with colored points according to the labels

using DataFrames
data_frame = DataFrame()
data_frame.x = x
data_frame.y = y
data_frame.cluster = labels

data_frame_clust = DataFrame()
data_frame_clust.x = x_clust
data_frame_clust.y = y_clust


using VegaLite
kmeans3_plot = data_frame |> @vlplot(:point, x=:x, y=:y, color=:cluster)
using FileIO
kmeans3_plot |> FileIO.save("/Users/Joel/Desktop/BIO397/week02.4/kmeans3_plot.png")

############################################################################

# now we can try with 4 clusters
k, cluster = ML.k_means_clustering(features,4)

# extract clusters
x_clust = k[:,1]
y_clust = k[:,2]

# extract labels
labels = cluster[:,3]
labels = string.(labels)

# plot again with colored points according to the labels

using DataFrames
data_frame = DataFrame()
data_frame.x = x
data_frame.y = y
data_frame.cluster = labels

data_frame_clust = DataFrame()
data_frame_clust.x = x_clust
data_frame_clust.y = y_clust


using VegaLite
kmeans4_plot = data_frame |> @vlplot(:point, x=:x, y=:y, color=:cluster)

# from looking at the data visually i think that it makes more sense in this
# case to choose 3 cluster centers.

############################################################################

