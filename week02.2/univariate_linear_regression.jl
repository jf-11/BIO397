
# UNIVARIATE LINEAR REGRESSION
############################################################################

using LinearAlgebra
using Random
using VegaLite
using DataFrames
using FileIO

############################################################################

# Data

trees = dataset("datasets", "trees");
x = trees[!, :Girth];
y = trees[!, :Height];

############################################################################

# Call the function

using ML
a = ML.linear_regression(x,y,0.001,10^6)[1]
b = ML.linear_regression(x,y,0.001,10^6)[2]
cost_change = ML.linear_regression(x,y,0.001,10^6)[3]

############################################################################

# PLOT IT

ypred = a .* x .+ b
d = DataFrame(:x => x, :y => y, :ypred => ypred);
univariate_plot = @vlplot(data=d)+
  @vlplot(
    mark = {:point, color=:black, filled=true},
    x = {:x, axis={title="x"}},
    y = {:y, axis={title="y"}}
  ) +
  @vlplot(
  	mark = {:line, color=:red},
  	x = :x,
  	y = :ypred
)

univariate_plot |> FileIO.save("/Users/Joel/Desktop/BIO397/week02.2/univariate_linreg_plot.png")

############################################################################

