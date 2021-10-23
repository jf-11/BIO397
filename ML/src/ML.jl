module ML
    using Statistics
    using StatsBase
    using Random
    include("linear_regression_functions.jl")
    include("multivariate_linear_regression_functions.jl")
    include("ridge_regression_functions.jl")
    include("logistic_regression_functions.jl")
    include("knn_functions.jl")
    include("k_mean_functions.jl")
end

