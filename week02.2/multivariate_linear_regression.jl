
# MULTIVARIATE LINEAR REGRESSION
############################################################################

using RDatasets
trees = dataset("datasets", "trees");
X = Matrix(trees[!, [:Girth,:Height]]);
Y = trees[!, :Volume];
A = ML.multivariate_linear_regression(X,Y,0.0001,10^6)

############################################################################
