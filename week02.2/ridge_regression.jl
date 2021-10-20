
# RIDGE REGRESSION
############################################################################

include("/Users/Joel/Desktop/BIO397/ML/ML.jl")
using RDatasets
trees = dataset("datasets", "trees");
x = trees[!, :Girth];
y = trees[!, :Height];

############################################################################