
# DATA PREPARATION
############################################################################

using DataFrames
using CSV
using Statistics
using DataFrames
using Distributions
include("/Users/Joel/Desktop/BIO397/final_project/data_prep_functions.jl")

# load the data
training_data = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/final_project/titanic_data/train.csv"))
test_data = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/final_project/titanic_data/test.csv"))

############################################################################

# have a look at the data
print(describe(training_data))
print(describe(test_data))

# Survived: Outcome of survival (0 = No; 1 = Yes)
# Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# Name: Name of passenger
# Sex: Sex of the passenger
# Age: Age of the passenger (Some entries contain NaN)
# SibSp: Number of siblings and spouses of the passenger aboard
# Parch: Number of parents and children of the passenger aboard
# Ticket: Ticket number of the passenger
# Fare: Fare paid by the passenger
# Cabin Cabin number of the passenger (Some entries contain NaN)
# Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

############################################################################

# age, cabin and embarked have missing values.
# how do we want to deal with them?

# in age we will replace missing values with the random values around the mean
# standard deviation. The two missing values in Embarked we will fill with the
# mode. In the cabing we are going to extract the deck and one hot encode it.

# which features do we want to use?
# PassenderId is not needed. We are going to drop Ticket as well, we cannot
# draw useful informations from it.

# how should we change the features?
# we are going to combine Sibpb and Parch to a new variable relatives. We have to
# scale the numerical variables. We are goind to extract titles from the names.

############################################################################

# DEALING WITH MISSING VALUES

# AGE
age = age_missing_vals2(training_data)
age_test = age_missing_vals2(test_data)

# EMBARKED
embarked = deepcopy(training_data.Embarked)
embarked = embarked_missing_vals(embarked)

embarked_test = deepcopy(test_data.Embarked)
embarked_test = embarked_missing_vals(embarked_test)

# CABIN
cabin = deepcopy(training_data.Cabin)
cabin = cabin_missing_vals(cabin)
false_vec_training = cabin.cabin_T .= 0
#cabin = cabin[:,Not(8)]

cabin_test = deepcopy(test_data.Cabin)
cabin_test = cabin_missing_vals(cabin_test)
cabin_test.cabin_T = false_vec_training[1:size(cabin_test,1)]

############################################################################

# TRANSFOMRING THE FEATURES

# ONE HOT ENCODE SEX
sex = deepcopy(training_data.Sex)
sex = sex_one_hot(sex)

sex_test = deepcopy(test_data.Sex)
sex_test = sex_one_hot(sex_test)

# COMBINE SIBSP AND PARCH
sibsp = deepcopy(training_data.SibSp)
parch = deepcopy(training_data.Parch)
relatives = sibsp_parch_combiner(sibsp,parch)
#large = relatives .> 6

sibsp_test = deepcopy(test_data.SibSp)
parch_test = deepcopy(test_data.Parch)
relatives_test = sibsp_parch_combiner(sibsp_test,parch_test)
#large_test = relatives_test .> 6

# EXTRACT THE TICKET LENGTHS
# ticket = ticket_number_length(training_data.Ticket)
# ticket_long = ticket .> 6
# ticket_short = ticket .< 3

# ticket_test = ticket_number_length(test_data.Ticket)
# ticket_long_test = ticket_test .> 6
# ticket_short_test = ticket_test .< 3

# SCALE THE NUMERICAL VARIABLES
age2 = deepcopy(age)
age_scaled = mean_normalization(age2)
relatives2 = deepcopy(relatives)
relatives_scaled = mean_normalization(relatives2)
fare = deepcopy(training_data.Fare)
fare_scaled = mean_normalization(fare)
pclass = deepcopy(training_data.Pclass)
pclass = pclass_one_hot(pclass)

age2_test = deepcopy(age_test)
age_scaled_test = mean_normalization(age2_test)
relatives2_test = deepcopy(relatives_test)
relatives_scaled_test = mean_normalization(relatives2_test)
fare_test = deepcopy(test_data.Fare)
fare_test = fare_missing_vals(fare_test)
fare_scaled_test = mean_normalization(fare_test)
pclass_test = deepcopy(test_data.Pclass)
pclass_test = pclass_one_hot(pclass_test)

# EXTRACT THE TITLE FROM THE NAME
name = deepcopy(training_data.Name)
name = title_extracter(name)
name_imp = name[:,1:4]
#name_rare = name[:,5] .+ name[:,6] .+ name[:,7] .+ name[:,8] .+ name[:,9] .+ name[:,10] .+ name[:,11] .+ name[:,12] .+ name[:,13] .+ name[:,14] .+ name[:,15] .+ name[:,16] .+ name[:,17]

name_test = deepcopy(test_data.Name)
name_test = title_extracter(name_test)
name_imp_test = name_test[:,1:4]
#name_rare_test = name_test[:,5] .+ name_test[:,6] .+ name_test[:,7] .+ name_test[:,8] .+ name_test[:,9]

############################################################################

# creating a new DataFrame
final_training_data = DataFrame()

final_training_data.Age = age_scaled
final_training_data.Relatives = relatives_scaled
final_training_data.Fare = fare_scaled

final_training_data = hcat(final_training_data,name_imp,sex,cabin,embarked,pclass)
final_training_data_labels = training_data.Survived


final_test_data = DataFrame()

final_test_data.Age = age_scaled_test
final_test_data.Relatives = relatives_scaled_test
final_test_data.Fare = fare_scaled_test

final_test_data = hcat(final_test_data,name_imp_test,sex_test,cabin_test,embarked_test,pclass_test)

# Bring the data frames in the same order
final_training_data2 = final_training_data[!,sort(names(final_training_data))]
final_test_data2 = final_test_data[!,sort(names(final_test_data))]

CSV.write("/Users/Joel/Desktop/BIO397/final_project/training_data_prepared.csv", final_training_data2)

# creating matrices
training_data_m = Matrix(final_training_data2)
test_data_m = Matrix(final_test_data2)

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

# TRAIN THE MODEL
############################################################################

using ScikitLearn
@sk_import linear_model: LogisticRegression
@sk_import neighbors: KNeighborsClassifier
@sk_import model_selection: StratifiedKFold
@sk_import model_selection: GridSearchCV
@sk_import metrics: accuracy_score
@sk_import ensemble: RandomForestClassifier

############################################################################

# LOGISTIC REGRESSION
function Log_Reg(X_train, y_train; nsplits=5, scoring="accuracy", n_jobs=1)
    model = LogisticRegression()
    parameters = Dict("penalty" => ("l1", "l2","elasticnet","none"))
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
    # train the model
    fit!(gridsearch,X_train,y_train)
    best_estimator = gridsearch.best_estimator_
    return best_estimator, gridsearch
end

best_est, grid = Log_Reg(training_data_m,final_training_data_labels)
pred = predict(best_est,test_data_m)

############################################################################

# KNN
function KNN_classification(X_train, y_train; nsplits=5, scoring="accuracy", n_jobs=1)
    model2 = KNeighborsClassifier()
    parameters = Dict("n_neighbors" => 16:2:18, "weights" => ("uniform", "distance"), "algorithm"=>("auto", "ball_tree", "kd_tree", "brute"))
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model2, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
    # train the model
    fit!(gridsearch,X_train,y_train)
    best_estimator = gridsearch.best_estimator_
    return best_estimator, gridsearch
end

best_est2, grid2 = KNN_classification(training_data_m,final_training_data_labels)
pred2 = predict(best_est2,test_data_m)

############################################################################

# RANDOM FOREST
function R_Forest(X_train, y_train; nsplits=5, scoring="accuracy", n_jobs=1)
    model3 = RandomForestClassifier()
    parameters = Dict("n_estimators" => 1:20:100)
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model3, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
    # train the model
    fit!(gridsearch,X_train,y_train)
    best_estimator = gridsearch.best_estimator_
    return best_estimator, gridsearch
end

best_est3, grid3 = R_Forest(training_data_m,final_training_data_labels)
pred3 = predict(best_est3,test_data_m)

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

# write to csv File
submission = DataFrame()
submission.PassengerId = test_data.PassengerId
submission.Survived = pred
CSV.write("/Users/Joel/Desktop/BIO397/final_project/logistics_regression.csv", submission)

submission2 = DataFrame()
submission2.PassengerId = test_data.PassengerId
submission2.Survived = pred2
CSV.write("/Users/Joel/Desktop/BIO397/final_project/knn.csv", submission2)

submission3 = DataFrame()
submission3.PassengerId = test_data.PassengerId
submission3.Survived = pred3
CSV.write("/Users/Joel/Desktop/BIO397/final_project/random_forest.csv", submission3)

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

## KNN 79.90 k = 16
## Logistic Regression 77.27
## Random Forest 74.63

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################