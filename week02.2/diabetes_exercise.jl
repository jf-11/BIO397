
# DIABETES DATA EXERCISE
############################################################################

using ML
using StatsBase
using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import datasets: load_diabetes
all_data = load_diabetes()
X = all_data["data"]
y = all_data["target"]
# age, sex, body mass index, average blood pressure, and six blood serum measurements
# n = 442 diabetes

Xt = fit(ZScoreTransform, X, scale=true)
X =StatsBase.transform(Xt, X)

############################################################################

# A

A = ML.ridge_linear_regression(X,y,0.0001,5,10^6)[1]
o = ones(size(X,1))
XX = hcat(o,X)
prediction = ML.ridge_predict(XX,A)
resid = y.-prediction

@sk_import linear_model: Ridge
model = Ridge(alpha=5)
ridge=fit!(model, X, y)
ypred = ridge.predict(X)
intercept,coef = ridge.intercept_, ridge.coef_
res = y.-ypred

# comparison of coef
coef
A

# the coefficients are pretty close to the built in function

############################################################################

# B
# find most important features
A_sorted = sort(A)
A

# bmi seems to be very important, sex is the least important according to
# the coeficient, serum5 measurement ist also importnant, bp seems to be
# important too
# --> bmi, s5, bp are the most important features

############################################################################