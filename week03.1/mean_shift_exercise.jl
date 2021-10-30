
# MEAN SHIFT EXERCISE
############################################################################

using ScikitLearn
@sk_import datasets: make_blobs
features, labels = make_blobs(n_samples=500, centers=3, cluster_std=0.55, random_state=0)

############################################################################

# plot the data with the true clusters, that we compare afterwards if our
# function worked.

using DataFrames
using VegaLite
data = DataFrame(:x => features[:,1], :y => features[:,2],:cluster=>string.(labels));
mean_shift_plot = @vlplot(data=data)+
  @vlplot(
    mark = {:point},
    color = :cluster,
    title = "Mean shift exercise, true clusters",
    x = {:x, axis={title="x"}},
    y = {:y, axis={title="y"}}
)

using FileIO
mean_shift_plot |> FileIO.save("/Users/Joel/Desktop/BIO397/week03.1/mean_shift_plot.png")

############################################################################

# now we can call our own function and check wether it matches the true
# clusters.

using ML
centroids = ML.mean_shift_function(features,3,0.8)
clusters = ML.cluster_assign_mean_shift(features,centroids)

############################################################################

# now we can plot the data
data2 = DataFrame(:x => features[:,1], :y => features[:,2],:cluster=>string.(clusters));
mean_shift_plot2 = @vlplot(data=data2)+
  @vlplot(
    mark = {:point},
    color = :cluster,
    title = "Mean shift exercise, own clusters",
    x = {:x, axis={title="x"}},
    y = {:y, axis={title="y"}}
)

using FileIO
mean_shift_plot2 |> FileIO.save("/Users/Joel/Desktop/BIO397/week03.1/mean_shift_plot_own_func.png")

############################################################################
