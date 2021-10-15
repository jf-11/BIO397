# Missing value.
# Mixed date formats.
# Different representations of the same values.
# Formulas (e.g. summation).
# Duplicate values.
# Mixed numerical scales.
# Redundant data.
# Spelling errors.

# US GOVERNMENT DATA

using CSV
using DataFrames
usgov_data = CSV.read("/Users/Joel/Desktop/BIO397/week01.2/dataset2.csv",DataFrame)

# missing values?
print(describe(usgov_data))
ismissing.(usgov_data)
# choose features that might be useful
new_data = DataFrame(project_id=usgov_data[!,6],lifecycle_cost=usgov_data[!,14],shedule_var=usgov_data[!,15],planned_cost=usgov_data[!,19])
print(describe(new_data))
# now we drop the missing values of project_id
updated_data = dropmissing(new_data,:project_id)
# to the other missing values we assign the means of the respective variables
using Statistics
mean_lifecycle_cost = mean(skipmissing(updated_data.lifecycle_cost))
mean_shedule_var = mean(skipmissing(updated_data.shedule_var))
mean_planned_cost = mean(skipmissing(updated_data.planned_cost))
# now assign them
updated_data2 = replace(updated_data.lifecycle_cost,missing=>mean_lifecycle_cost)
updated_data3 = replace(updated_data.shedule_var,missing=>mean_shedule_var)
updated_data4 = replace(updated_data.planned_cost,missing=>mean_planned_cost)
# create new data DataFrame
nomissing_data = DataFrame(project_id=updated_data.project_id,lifecycle_cost=updated_data2,
shedule_var=updated_data3,planned_cost=updated_data4)
# check if it worked:
ismissing.(nomissing_data)
# it worked...

# Mixed date formats.
# We have no date formats in our data frame anymore...

# Different representations of the same values.
# We have no different representations of the same values in our data frame...

# Formulas (e.g. summation).
# We have also no formulas but we used the mean of the variables if the value was missing.

# Duplicate values.
findall(nonunique(nomissing_data))
# We can see that we have no duplicate values in our DataFrame.

# Mixed numerical scales.
# We need to standardize our data:
sdf(x, m, s) = (x-m)/s

sdf_lifecycle = sdf.(nomissing_data.lifecycle_cost, mean(nomissing_data.lifecycle_cost), std(nomissing_data.lifecycle_cost))
sdf_shedule_var = sdf.(nomissing_data.shedule_var, mean(nomissing_data.shedule_var), std(nomissing_data.shedule_var))
sdf_planned_cost = sdf.(nomissing_data.planned_cost, mean(nomissing_data.planned_cost), std(nomissing_data.planned_cost))

sd_data = DataFrame(project_id=nomissing_data.project_id,lifecycle_cost=sdf_planned_cost,shedule_var=sdf_shedule_var,planned_cost=sdf_planned_cost)

# Redundant data.
# We checked our features before that we have no rdundant data.

# Spelling errors.
names(sd_data)
# we have no spelling errors in our columns names. Beside of that we do not have any characters.

# Now we can save the file:
CSV.write("usgov_clean_data.csv", sd_data)