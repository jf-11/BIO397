
# RIDGE REGRESSION
############################################################################

using ML
using RDatasets
trees = dataset("datasets", "trees");
X = Matrix(trees[!, [:Girth,:Height]]);
Y = trees[!, :Volume];

############################################################################

# A
A = ML.ridge_linear_regression(X,Y,0.0001,0.1,10^7)[1]
cost = ML.ridge_linear_regression(X,Y,0.0001,0.1,10^7)[2]


############################################################################

# B
# 80% of data --> approx 25 datapoints
using StatsBase
train_ind = StatsBase.sample(1:size(X,1),25,replace=false)
train = X[train_ind,:]
test = X[Not(train_ind),:]
# fit with training data
A2 = ML.ridge_linear_regression(train,Y[train_ind],0.0001,0.1,10^7)[1]
# test with test data
o = ones(size(test,1))
XX = hcat(o,test)
prediction = ML.ridge_predict(XX,A2)
# compare to the real data
comparison = hcat(Y[Not(train_ind)],prediction)
residuals = Y[Not(train_ind)] .- prediction
print(residuals)
# We can see that our model isn't too far of the real values

############################################################################

# C
# fit higher polynomial model
iris = dataset("datasets", "iris");
X_iris = Matrix(iris[!, [:SepalLength,:SepalWidth,:PetalLength]]);
Y_iris = iris[!, :PetalWidth];
X_iris_new = hcat(X_iris[:,1],X_iris[:,2].^2,X_iris[:,3].^3)
A3 = ML.ridge_linear_regression(X_iris_new,Y_iris,0.0001,0.1,10^5)[1]
o = ones(size(X_iris,1))
XX_iris = hcat(o,X_iris)
prediction_iris = ML.ridge_predict(XX_iris,A3)
comparison = hcat(Y_iris,prediction_iris)
residuals_iris = Y_iris .- prediction_iris

using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: Ridge
model = Ridge(alpha=0.1)
ridge=fit!(model, X_iris_new, Y_iris)
ypred = ridge.predict(X_iris)
coef = ridge.intercept_, ridge.coef_
print(coef)
print(A)

############################################################################

# D
# Comparison of tree model with ScikitLearn built in

using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: Ridge
model = Ridge(alpha=0.1)
ridge=fit!(model, X, Y)
ypred = ridge.predict(X)
coef = ridge.intercept_, ridge.coef_
print(coef)
print(A)
# The intercept is different, the coeficients for Grith and Height
# aren't too far off.

# (-57.99878658933007, [4.706019761728996, 0.33977082687908877])
# [-38.78932302181156, 4.817204633265637, 0.06927958107882684]

############################################################################
