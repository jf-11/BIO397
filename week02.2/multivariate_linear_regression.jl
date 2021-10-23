
# MULTIVARIATE LINEAR REGRESSION
############################################################################

using ML
using RDatasets
trees = dataset("datasets", "trees");
X = Matrix(trees[!, [:Girth,:Height]]);
Y = trees[!, :Volume];
A = ML.multivariate_linear_regression(X,Y,0.0001,10^6)[1]
cost = ML.multivariate_linear_regression(X,Y,0.0001,10^6)[2]

############################################################################

o = ones(size(X,1))
XX = hcat(o,X)
pred = ML.multivariate_predict(XX,A)