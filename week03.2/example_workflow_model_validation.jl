
# EXAMPLE WORKFLOW MODEL VALIDATION
############################################################################

using ScikitLearn
@sk_import neighbors: KNeighborsClassifier
@sk_import model_selection: StratifiedKFold
@sk_import metrics: f1_score
@sk_import datasets: load_breast_cancer
@sk_import model_selection: GridSearchCV
@sk_import preprocessing: PolynomialFeatures
@sk_import preprocessing: RobustScaler
@sk_import model_selection: train_test_split

############################################################################

all_data = load_breast_cancer()
X = all_data["data"];
y = all_data["target"];

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

############################################################################

"""
Add polynomials to features to your data.
"""
function addpolynomials(X; degree=2, interaction_only=false, include_bias=true)
  poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
  fit!(poly, X)
  x2 = transform(poly, X)
  return x2
end

############################################################################

function KNN_classification(X_train, y_train; nsplits=5, scoring="f1", n_jobs=1, stratify=nothing)

# Scale the data first
  rscale = RobustScaler()
  X_train = rscale.fit_transform(X_train)

  # build the model and grid search object
  model = KNeighborsClassifier()
  parameters = Dict("n_neighbors" => 1:2:30, "weights" => ("uniform", "distance"))
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

# Make predictions:

# transform X_test
X_test_transformed = rscale.transform(X_test)

y_pred = predict(best_estimator, X_test_transformed)

# Evaluate the predictions

f1_score(y_test, y_pred)

## learning for model evaluation

@sk_import model_selection: learning_curve

cv = StratifiedKFold(n_splits=nsplits, shuffle=true)
train_sizes, train_scores, test_scores = learning_curve(best_estimator, X_train, y_train, cv=cv, scoring="accuracy", shuffle=true)

############################################################################
