
# LeNet-5 model with Flux
############################################################################

using Flux, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using MLDatasets
using ScikitLearn
using Statistics: mean

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

############################################################################

# define parameters and image size
α = 0.001
iterations = 10^4
imgsize = (28,28,1)

# change the dimensions of the images
x_train_reshaped = reshape(train_x,28,28,1,60000)

# build the model
model = Chain(
    Conv((5, 5), 1=>6,pad=(2,2), relu),
    MeanPool((2,2)),
    Conv((5, 5), 6=>16, relu),
    MeanPool((2,2)),
    flatten,
    Dense(400,120,relu),
    Dense(120,84,relu),
    Dense(84,10),
    softmax
    )

# define the cost function
cost(x, y) = logitcrossentropy(model(x), y)

# define the optimiser function
optimiser = Descent(0.005)

############################################################################

# train the model
data = [train_x,train_y]
for i in iterations
    Flux.train!(cost(x_train_reshaped,y), params(model), data, optimiser)
end

############################################################################

# test the model
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
acc = accuracy(test_x, test_y)
print(acc)

############################################################################