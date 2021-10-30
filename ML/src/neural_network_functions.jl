
# NEURAL NETWORK FUNCTIONS
############################################################################

# write two activation functions relu
# relu for inner layer
function neural_relu(z)
    return max(0,z)
end

# logistic for output layer
function neural_logistic(z)
    predict = max(min(1/(1 + exp(-z)), 0.9999), 0.001)
    return predict
end

############################################################################

# write linear line function
function neural_linear(X,W,b)
    lin = W * X .+ b
    return lin
end

############################################################################

# write a function called model
function neural_model(X, W1, b1, W2, b2, W3, b3)
    first_layer = neural_linear(X,W1,b1)
    first_layer_act = neural_relu.(first_layer)
    second_layer = neural_linear(first_layer_act,W2,b2)
    second_layer_act = neural_relu.(second_layer)
    output_layer = neural_linear(second_layer_act,W3,b3)
    output = neural_logistic.(output_layer)
    return output
end

############################################################################

# write a cost function
function neural_cost(y_actual::Real,y_pred::Real)
    cost = -y_actual*log10(y_pred)-(1-y_actual)*log10(1-y_pred)
    return cost
end

############################################################################

# cost function that calculates the average
function neural_cost_final(X, Y, W1, b1, W2, b2, W3, b3)
    y_pred= neural_model(X, W1, b1, W2, b2, W3, b3)
    cost= neural_cost.(Y,y_pred)
    cost_average= mean(cost)
    return cost_average
end

############################################################################

# gradient descent function
function neural_gradient_descent(X, Y, W1, b1, W2, b2, W3, b3,α)
    grads = gradient(() -> neural_cost_final(X, Y, W1, b1, W2, b2, W3, b3), params(W1, b1, W2, b2, W3, b3))
    W1_new= W1 .- α .* grads[W1]
    W2_new= W2 .- α .* grads[W2]
    W3_new= W3 .- α .* grads[W3]
    b1_new= b1 .- α .* grads[b1]
    b2_new= b2 .- α .* grads[b2]
    b3_new= b3 .- α .* grads[b3]
    return W1_new, b1_new, W2_new, b2_new, W3_new, b3_new
end

############################################################################

function neural_network_function(X, Y, α, iterations)
    W1 = 0.01 .* randn(20,size(X,1))
    W2 = 0.01 .* randn(15,size(W1,1))
    W3 = 0.01 .* randn(10,size(W2,1))
    b1 = zeros(size(W1,1),1)
    b2 = zeros(size(W2,1),1)
    b3 = zeros(size(W3,1),1)
    for i in 1:iterations
        W1, b1, W2, b2, W3, b3 = neural_gradient_descent(X, Y, W1, b1, W2, b2, W3, b3, α)
    end
    return W1,b1,W2,b2,W3,b3
end

############################################################################


