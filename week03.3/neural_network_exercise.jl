
# NEURAL NETWORK EXERCISE
############################################################################

using Flux: onehotbatch, onecold
using ScikitLearn
using Statistics: mean
using Random
using ML

############################################################################

#load the data
@sk_import datasets: load_digits
@sk_import model_selection: train_test_split
digits = load_digits();
X = Float32.(digits["data"]);  # make the X Float32 to save memory
y = digits["target"];
Y = Int32.(onehotbatch(y, 0:9));
nsamples, nfeatures = size(X)

# split the data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y', train_size=0.8)

X_train = X_train'
X_test = X_test'
Y_train = Y_train'
Y_test = Y_test'

############################################################################

W1,b1,W2,b2,W3,b3 = ML.neural_network_function(X_train, Y_train, 0.1, 10^4)

ypred = ML.neural_model(X_test, W1, b1, W2, b2, W3, b3)
Y_test = onecold(Y_test)
ypred = onecold(ypred)

accuracy = count(Y_test .== ypred)/length(Y_test)
# accuracy = 0.9555555555555556

############################################################################