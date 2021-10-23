
# LOGISTIC REGRESSION FUNCTIONS
############################################################################

function logistic_regression_predict(X,A)
    z = X*A
    pred = @. 1 / (1 + exp(-(z)))
    return pred
end

############################################################################

function logistic_regression_gradient(X,y,A,α::Number)
    n=size(X,1)
    pred = logistic_regression_predict(X,A)
    a_gradient = (1/n) * (-1) * ((y.-pred)')*X
    A_new = A .- α * a_gradient'
    return A_new
end

############################################################################

function logistic_regression(X,y,α::Number,iterations::Number)
    o = ones(size(X,1))
    X2 = hcat(o,X)
    A = zeros(size(X2,2))
    for i in 1:iterations
        A = logistic_regression_gradient(X2,y,A,α)
    end
    return A
end

############################################################################

