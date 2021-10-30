
# DECISION TREE PLOT
############################################################################

using ScikitLearn
@sk_import tree: DecisionTreeClassifier

############################################################################

# Load the data
breast_cancer_data = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/week03.2/Breast_cancer_data.csv"))
y = breast_cancer_data.diagnosis
X = breast_cancer_data[!,Not(:diagnosis)]
X = Matrix(X)

############################################################################

# Create a tree
model = DecisionTreeClassifier(min_samples_split=2, random_state=0, min_samples_leaf=1, max_depth=4)
fit!(model, X, y)

model.feature_importances_
# 0.004532369758348476
# 0.08279255117221247
# 0.09775555736816893
# 0.7241904416671319
# 0.0907290800341383

# Plot the tree
using PyCall
@sk_import tree: export_graphviz
export_graphviz(model, out_file="mytree", class_names=["Healthy", "Cancerous"], feature_names=["means_radius","mean_texture","mean_perimeter","mean_area",], leaves_parallel=true, impurity=false, rounded=true, filled=true, label="root", proportion=true)


# The `export_graphviz` function creates a file named "mytree" in the same directory you ran it.
# Copy the content of the file and paste them in this website to view your tree:
# https://graphviz.christine.website/ or https://edotor.net/

############################################################################
