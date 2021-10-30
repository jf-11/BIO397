
# ANOMALY DETECTION EXERCISE 2
############################################################################

# load the data
using CSV
using DataFrames
df = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/week03.4/creditcardfraud_normalised.csv"))

############################################################################

# test the function
using ML
data = Matrix(df)
outliers_index,outliers_encoding = ML.outlier_detecter(data,0.05)

############################################################################