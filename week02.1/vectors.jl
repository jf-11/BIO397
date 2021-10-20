
# VECTORS
############################################################################

# y = ax + b with slope a and intercept b

using VegaDatasets
using DataFrames
using Plots
using PlotThemes
car = DataFrame(dataset("cars"));

function predict(x::Real,a::Real,b::Real)
    return x::Real*a + b
end

y_predicted = predict.(car.Weight_in_lbs,0.1,-100)

theme(:ggplot2;)
scatter(car.Weight_in_lbs,car.Displacement,title="Weight vs. Displacement",label=false)
plot!(car.Weight_in_lbs,y_predicted,label = "y = ax + b")
xlabel!("Weight (lbs)")
ylabel!("Displacement")
savefig("plot.png")