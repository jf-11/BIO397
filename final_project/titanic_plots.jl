
########################################################################################################

using Plotly
using FileIO
training_data = DataFrame(CSV.File("/Users/Joel/Desktop/BIO397/final_project/titanic_data/train.csv"))

########################################################################################################

age_plot = plot(training_data, x=:Age, kind="histogram", color=:Survived, Layout(barmode="stack"))
fare_plot = plot(training_data,x=:PassengerID, y=:Fare,color=:Survived,kind="scatter", mode="markers")
training_data.relatives = training_data.Parch .+ training_data.SibSp
relatives_plot = plot(training_data,x=:relatives,color=:Survived,kind="histogram", mode="markers")
x = names(name)
y = []
for i in 1:size(name,2)
    push!(y,sum(name[:,i]))
end
name_df = DataFrame()
name_df.names = x
name_df.values = y
names_plot = plot(name_df,x=:names, y=:values,kind="bar", mode="markers")

########################################################################################################

savefig(age_plot,"age_plot.png")
savefig(fare_plot,"fare_plot.png")
savefig(relatives_plot,"relatives_plot.png")
savefig(names_plot,"names_plot.png")

########################################################################################################