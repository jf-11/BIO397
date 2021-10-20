
# LINEAR REGRESSION FUNCTIONS
############################################################################

using LinearAlgebra

############################################################################

function univariate_predict(x::Number,a::Number,b::Number)
    pred = a * x + b
    return pred
end

############################################################################

function univariate_gradient(X::Array,Y::Array,a::Number,b::Number,α::Number)
    pred = univariate_predict.(X,a,b)
    b_gradient = (1/length(X)) * sum(-(Y .- pred))
    a_gradient = (1/length(X)) * sum(-(Y .- pred).*X)
    a_new = a - α * a_gradient
    b_new = b - α * b_gradient
    return a_new,b_new
end

############################################################################

function univariate_cost(X::Array,Y::Array,a::Number,b::Number)
    pred = univariate_predict.(X,a,b)
    cost = (1/2*length(X))*(sum((Y.-pred).^2))
    return cost
end

############################################################################

function linear_regression(X::Array,Y::Array,α::Number,iterations::Number)
    a = 0
    b = 0
    cost = []
    for i in 1:iterations
        ncost = univariate_cost(X,Y,a,b)
        a, b = univariate_gradient(X,Y,a,b,α)
        push!(cost,ncost-univariate_cost(X,Y,a,b))
    end
    return a,b,cost
end

############################################################################

