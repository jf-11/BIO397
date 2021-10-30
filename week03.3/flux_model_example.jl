
# NEURAL NETWORK EXAMPLE FLUX
############################################################################

using Flux
using Flux: onecold, crossentropy, throttle, onehotbatch
using ScikitLearn
using Base.Iterators: repeated
using StatsBase: sample
@sk_import datasets: load_digits
@sk_import model_selection: train_test_split
digits = load_digits();
X = Float32.(transpose(digits["data"]));  # make the X Float32 to save memory
y = digits["target"];
Y = Int32.(onehotbatch(y, 0:9));
nfeatures, nsamples = size(X)

############################################################################

## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X', y, train_size=0.80, stratify=y)
X_train = X_train'
X_test = X_test'

## One hot encode y
Y_train = onehotbatch(y_train, 0:9)
Y_test = onehotbatch(y_test, 0:9)

############################################################################

## Build a network. The output layer uses softmax activation, which suits multiclass classification problems.
model = Chain(
    Dense(nfeatures, 20, Flux.relu),
    Dense(20, 15, Flux.relu),
    Dense(15, 10),
    softmax
)

## cross entropy cost function
cost(x, y) = crossentropy(model(x), y)

opt = Descent(0.005)  # Choose gradient descent optimizer with alpha=0.005
dataset = repeated((X_train, Y_train), 2000)  # repeat the dataset 2000 times, equivalent to running 2000 iterations of gradient descent
Flux.train!(cost, params(model), dataset, opt)

############################################################################

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
accuracy(X_test, Y_test)

############################################################################