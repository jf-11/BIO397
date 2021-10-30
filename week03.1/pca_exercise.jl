
# PCA EXERCISE
############################################################################

# load the dat
using ScikitLearn
using ML
@sk_import datasets: load_breast_cancer
all_data = load_breast_cancer()

X = all_data["data"]
y = all_data["target"]
Y = string.(y)

############################################################################

# normalize
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

# run our pca function
X_new, total_explained_variance = ML.pca_function(X_normalized,2)
print(total_explained_variance)
# 0.7038117901347681 of the variance is explained by these two principal components

############################################################################

# make the plot
using Gadfly
layer1 = layer(x=X_new[:,1], y=X_new[:,2],color=Y,Geom.point)

Gadfly.with_theme(:dark) do
pca_plot = plot(layer1,Guide.colorkey(title="Diagnosis"),
Guide.title("Breast cancer data"))
end

using Cairo, Fontconfig
draw(PNG("pca_plot.png"), pca_plot)

############################################################################

# Exercise 2 reduce to k dimensions that new k features explain 99% of
# the variance

for i in 1:size(X,2)
    X_reduced, explained_var = ML.pca_function(X_normalized,i)
    if explained_var > 0.99
        print([i,explained_var])
        break
    end
end

# [16.0, 0.9903663100977914]
# we need k=16 that 99% of the variance can be explained...

X_reduced, explained_var = ML.pca_function(X_normalized,16)
print(100*explained_var)

############################################################################

# check our result with the built in scikit library
@sk_import decomposition: PCA
import ScikitLearn: fit!, predict
model = PCA(2)
model.fit(X_normalized)
sum(model.explained_variance_ratio_)
# 0.7038117901347671 built in
# 0.7038117901347681 my function
# the results are the same.

############################################################################