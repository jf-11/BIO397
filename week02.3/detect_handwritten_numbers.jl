
# DETECT HAND-WRITTEN NUMBERS
############################################################################

using ML
using ScikitLearn
@sk_import datasets: load_digits
digits = load_digits();
X = digits["data"];
y = digits["target"]

############################################################################

# CREATE LIST OF Y_
list_y = []
y_0 = y .==0
push!(list_y,y_0)
for i in 1:9
    y_i = y.==i
    push!(list_y,y_i)
end

# CREATE MATRIX CONTAINING ALL COEFFICIENTS
matrix = 1:65
for i in list_y
    parameters = ML.logistic_regression(X,i,0.001,10^5)
    matrix = hcat(matrix,parameters)
end

# GET FINAL COEF MATRIX
coef_matrix = matrix[1:end, 1:end .!= 1]

# PREPARE X2 FOR PREDICTION
o = ones(size(X,1))
X2 = hcat(o,X)

# DO PREDICTION
prediction = 1:1797
for i in 1:size(coef_matrix,2)
    prediction = hcat(prediction,ML.logistic_regression_predict(X2,coef_matrix[:, i]))
end

# GETTING FINAL PREDICTION MATRIX
prediction = prediction[1:end, 1:end .!= 1]

############################################################################

# GETTING HIGHEST VALUE OUT OF PREDICTION MATRIX

highest = findmax(prediction,dims=2)[2]

# EXTRACTING PREDICTIONS
preds = getindex.(getproperty.(highest,:I),2).-1

# CREATE A FINAL TABLE
final = hcat(y,preds)

# CALCULATION ACCURACY
count(preds.==y)/length(y) # 0.99054
count(preds.!=y) # 17

############################################################################


