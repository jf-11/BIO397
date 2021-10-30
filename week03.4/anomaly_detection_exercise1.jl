
# ANOMALY DETECTION EXERCISE 1
############################################################################

# load the data
using ScikitLearn
using VegaLite
@sk_import covariance: EllipticEnvelope
@sk_import datasets: make_moons
@sk_import datasets: make_blobs

# Example settings
n_samples = 300
outliers_fraction = 0.05
n_outliers = round(Int64, outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

X, y = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, random_state=1, n_samples=n_inliers, n_features=2)

@vlplot(mark=:point, x=X[:, 1], y=X[:, 2])

############################################################################

# now we can test the function
using ML
outliers_index,encoding = ML.outlier_detecter(X,0.05)

############################################################################

# lets plot the data
@vlplot(mark=:point, x=X[:, 1], y=X[:, 2],color=encoding)

############################################################################