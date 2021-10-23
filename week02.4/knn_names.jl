
# KNN NAMES
############################################################################

using ML
using StatsBase
using Random
using DataFrames
using CSV

############################################################################

# importing the file
data= DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/week02.4/name_gender.txt", header= false))

############################################################################

# splitting in training and testing set
# shuffle
shuffled = shuffle(1:size(data,1))
ind = shuffled[1:2493]
ind2 = shuffled[2494:end]
# get predefined data
defined_names = data[ind,:]
# get names to test
test_names = data[ind2,1]
# get labels to check
test_labels = data[ind2,2]

############################################################################

# test the function

my_name_prediction = ML.predict_gender(defined_names,"joel",3)
# male --> correct prediction

genders = []
for i in test_names
    push!(genders,ML.predict_gender(defined_names,i,3))
end

comparison = genders .== test_labels
total = sum(comparison) # 484

total / length(test_labels) # 0.777

# test with another K

genders2 = []
for i in test_names
    push!(genders2,ML.predict_gender(defined_names,i,5))
end

comparison2 = genders2 .== test_labels
total2 = sum(comparison)

total2 / length(test_labels)

############################################################################