
# FUNCTIONS FOR MULTIVARIATE CASE

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
    pred = multivariate_predict(X,A)
    cost = (1/2*size(X,1)) * sum((pred.-Y).^2)
    return cost
end

############################################################################

function multivariate_linear_regression(X,Y,α::Number,iterations::Number)
    o = ones(size(X,1))
    X2 = hcat(o,X)
    A = zeros(size(X2,2))
    cost = []
    for i in 1:iterations
        ncost = multivariate_cost(X2,Y,A)
        A .= multivariate_gradient(X2,Y,A,α)
        push!(cost,ncost-multivariate_cost(X2,Y,A))
    end
    return A, cost
end

############################################################################