
# RIDGE REGRESSION FUNCTIONS
############################################################################

function ridge_predict(X,A)
    pred = X*A
    return pred
end

############################################################################

function ridge_gradient(X,Y,A,α::Number,λ)
    a_gradient = (1/size(X,1)) * ((X*A-Y)')*X .+ (λ/size(X,1).*A)'
    A_new = A .- α * a_gradient'
    return A_new
end

############################################################################

function ridge_cost(X,Y,A,λ)
    pred = ridge_predict(X,A)
    cost = (1/2*size(X,1)) * sum((pred.-Y).^2) + λ * sum(A.^2)
    return cost
end

############################################################################

function ridge_linear_regression(X,Y,α::Number,λ,iterations::Number)
    o = ones(size(X,1))
    X2 = hcat(o,X)
    A = zeros(size(X2,2))
    λ = λ
    cost = []
    for i in 1:iterations
        ncost = ridge_cost(X2,Y,A,λ)
        A .= vec(ridge_gradient(X2,Y,A,α,λ))
        push!(cost,ncost-ridge_cost(X2,Y,A,λ))
    end
    return A,cost
end

############################################################################
