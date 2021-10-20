
# MONTE CARLO REGRESSION
############################################################################

using Random
using VegaLite
using DataFrames
using FileIO

############################################################################

x = collect(1:10) .+ randn(10)
y = collect(1:10) .+ randn(10)

@vlplot(
  mark = {:point, color=:black, filled=true},
  x = {x, axis={title="Independent variable"}},
y = {y, axis={title="Dependent variable"}}
)

############################################################################

# Loss function 2
function mc_regressionL2(x,y)
    best_a,best_b = randn(2)
    best_fit = @. sum((y-(best_a*x-best_b))^2)
    for i in 1:10^6
        a,b = randn(2)
        loss = @. sum((y-(a*x-b))^2)
        if loss < best_fit
            best_a = a
            best_b = b
        end
    end
    return best_a, best_b
end

# Loss function 1
function mc_regressionL1(x,y)
  best_a,best_b = randn(2)
  best_fit = @. sum((y-(best_a*x-best_b))^2)
  for i in 1:10^6
      a,b = randn(2)
      loss = @. sum(abs(y-(a*x-b)))
      if loss < best_fit
          best_a = a
          best_b = b
      end
  end
  return best_a, best_b
end

############################################################################

a,b = mc_regressionL2(x,y)
a,b = mc_regressionL1(x,y)

############################################################################

y_pred = @. a*x+b
d = DataFrame(:x => x, :y => y, :ypred => y_pred);
p1 = @vlplot(data=d)+
  @vlplot(
    mark = {:point, color=:black, filled=true},
    x = {:x, axis={title="Independent variable"}},
    y = {:y, axis={title="Dependent variable"}}
  ) +
  @vlplot(
  	mark = {:line, color=:red},
  	x = :x,
  	y = :ypred
)

p1 |> FileIO.save("/Users/Joel/Desktop/BIO397/week02.2/mc_lin_reg.png")

############################################################################

# COMPARING THE ALGORITHM TO THE BUILT IN FUNCTION

using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: LinearRegression
model = LinearRegression()
dd = DataFrame(:x => x, :y => y);
x = reshape(x, 10, 1)  # needed for the fit! function
fit!(model, x, y)
ypred = predict(model, x)
dd.ypred = ypred
intercept, coef = model.intercept_, model.coef_[1]

# plotting
p2 = @vlplot(data=dd)+
  @vlplot(
    mark = {:point, color=:black, filled=true},
    x = {:x, axis={title="Independent variable"}},
    y = {:y, axis={title="Dependent variable"}}
  ) +
  @vlplot(
  	mark = {:line, color=:red},
  	x = :x,
  	y = :ypred
)

p2 |> FileIO.save("/Users/Joel/Desktop/BIO397/week02.2/scikit_lin_reg.png")