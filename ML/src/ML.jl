module ML
    using Statistics
    using StatsBase
    using Random
    using LinearAlgebra
    using Flux
    using Distributions
    include("linear_regression_functions.jl")
    include("multivariate_linear_regression_functions.jl")
    include("ridge_regression_functions.jl")
    include("logistic_regression_functions.jl")
    include("knn_functions.jl")
    include("k_mean_functions.jl")
    include("pca_functions.jl")
    include("mean_shift_functions.jl")
    include("neural_network_functions.jl")
    include("lasso_regression_functions.jl")
    include("anomaly_detection_function.jl")
end

