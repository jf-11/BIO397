
# ANALYTICAL SOLUTION
############################################################################

using Random
using VegaLite
using DataFrames
using FileIO
using RDatasets
using Statistics

############################################################################

function analytical_lin_reg(x,y)
    a = sum((x .* y - mean(y).*x))/sum((x.^2 - mean(x).*x))
    b = mean(y) - a * mean(x)
    return a,b
end

############################################################################

trees = dataset("datasets", "trees");
x = trees[!, :Girth];
y = trees[!, :Height];

############################################################################

# call the function
a,b = analytical_lin_reg(x,y)

############################################################################

# PLOT

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
