
# EXERCISE MODEL VALIDATION
############################################################################

using DataFrames
using CSV
using ScikitLearn
@sk_import neighbors: KNeighborsClassifier
@sk_import model_selection: StratifiedKFold
@sk_import metrics: f1_score
@sk_import model_selection: GridSearchCV
@sk_import preprocessing: RobustScaler
@sk_import model_selection: train_test_split

############################################################################

# import the data
breast_cancer_data = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/week03.2/Breast_cancer_data.csv"))

############################################################################

# split the data into training and test set
diagnosis = breast_cancer_data.diagnosis
X = breast_cancer_data[!,Not(:diagnosis)]
X = Matrix(X)
X_train, X_test, y_train, y_test  = train_test_split(X,diagnosis,train_size=0.8)

############################################################################

# define function
# we do here KMeans instead of knn...
function KNN_classification(X_train, y_train; nsplits=5, scoring="f1", n_jobs=1, stratify=nothing)
    # Scale the data first
    rscale = RobustScaler()
    X_train = rscale.fit_transform(X_train)
    # build the model and grid search object
    model = KNeighborsClassifier()
    parameters = Dict("n_neighbors" => 1:2:40, "weights" => ("uniform", "distance"))
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
    # train the model
    fit!(gridsearch, X_train, y_train)
    best_estimator = gridsearch.best_estimator_
    return best_estimator, gridsearch, rscale
    end

############################################################################

# Use the function
best_estimator, gridsearch, rscale = KNN_classification(X_train, y_train)
print(best_estimator) # KNeighborsClassifier(n_neighbors=23)

############################################################################

# Make predictions:
# transform X_test
X_test_transformed = rscale.transform(X_test)

y_pred = predict(best_estimator, X_test_transformed)

############################################################################

# Evaluate the predictions
f1_score(y_test, y_pred)
# 0.9736842105263158

############################################################################

# learning for model evaluation
@sk_import model_selection: learning_curve
nsplits = 5
cv = StratifiedKFold(n_splits=nsplits, shuffle=true)
train_sizes, train_scores, test_scores = learning_curve(best_estimator, X_train, y_train, cv=cv, scoring="accuracy", shuffle=true)

############################################################################

