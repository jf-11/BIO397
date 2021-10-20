
# FUNCTIONS FOR MULTIVARIATE CASE

using LinearAlgebra

############################################################################

function multivariate_predict(X,A)
    pred = X*A
    return pred
end

############################################################################

function multivariate_gradient(X,Y,A,α::Number)
    a_gradient = (1/size(X,1)) * ((X*A-Y)')*X
    A_new = A - α * a_gradient'
    return A_new
end

############################################################################

function multivariate_cost(X,Y,A)
    pred = multivariate_predict.(X,A)
    cost = (1/2*size(X,1)) * (pred-Y)^2
    return cost
end

############################################################################

function multivariate_linear_regression(X,Y,α::Number,iterations::Number)
    o = ones(size(X,1))
    X2 = hcat(o,X)
    A = zeros(size(X2,2))
    for i in 1:iterations
        A .= multivariate_gradient(X2,Y,A,α)
    end
    return A
end

############################################################################