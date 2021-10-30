
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

# normalize the data
using Statistics
X_normalized = zeros(size(X))
for i in 1:size(X,2)
    mm = mean(X[:,i])
    sd = std(X[:,i])
    for p in 1:size(X,1)
        X_normalized[p,i] = (X[p,i] - mm) / (maximum(X[:,i])-minimum(X[:,i]))
    end
end

############################################################################

# A

A = ML.ridge_linear_regression(X_normalized,y,0.0001,5,10^6)[1]
o = ones(size(X_normalized,1))
XX = hcat(o,X_normalized)
prediction = ML.ridge_predict(XX,A)
resid = y.-prediction

@sk_import linear_model: Ridge
model = Ridge(alpha=5)
ridge=fit!(model, X_normalized, y)
ypred = ridge.predict(X)
intercept,coef = ridge.intercept_, ridge.coef_
res = y.-ypred

# comparison of coef
coef
A

# the coefficients are pretty close to the built in function

# 3.9088529457930266
# -19.741494035599406
# 108.0700442610079
#  68.88387087144218
# -11.886693155391033
# -17.545300232908268
# -50.98236821771218
#  34.63319875268266
#  94.05591375271841
#  28.48249881515246

# 4.057996942576838
# -19.93033663064575
# 106.751119036915
#  69.17055821369193
#  -10.5789711987264
# -51.798593836543695
#  35.552433112218225
#  91.50889819045086
#  30.00627513318359

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