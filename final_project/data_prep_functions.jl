
# DATA PREPARATION FUNCTIONS
############################################################################

# DEAL WITH MISSING VALUES
# age
function age_missing_vals(age_col)
    mm = mean(skipmissing(age_col))
    sd = std(skipmissing(age_col))
    object = Normal(mm,sd)
    age_col[ismissing.(age_col)] = rand(object,length(age_col[ismissing.(age_col)]))
    return age_col
end

function age_missing_vals2(data)
    male1= []
    male2= []
    male3= []
    female1=[]
    female2=[]
    female3=[]
    age_col= data[!,:Age]
    for (i,sex) in enumerate(data[!,:Sex])
        if ismissing(data[i,:Age])
            continue
        elseif sex== "male" && data[i,:Pclass]==1
            push!(male1,data[i,:Age])
        elseif sex== "male" && data[i,:Pclass]==2
            push!(male2,data[i,:Age])
        elseif sex== "male" && data[i,:Pclass]==3
            push!(male3,data[i,:Age])
        elseif sex== "female" && data[i,:Pclass]==1
            push!(female1,data[i,:Age])
        elseif sex == "female" && data[i,:Pclass]==2
            push!(female2,data[i,:Age])
        elseif sex == "female" && data[i,:Pclass]==3
            push!(female3,data[i,:Age])
        end
    end
    for (i,sex) in enumerate(data[!,:Sex])
        if ismissing(data[i,:Age]) && sex== "male" && data[i,:Pclass]==1
            sd= std(male1)
            mm = mean(male1)
            object = Normal(mean(male1),sd)
            age_col[i] = rand(object)
        elseif ismissing(data[i,:Age]) && sex== "male" && data[i,:Pclass]==2
            sd= std(male2)
            mm = mean(male2)
            object = Normal(mean(male2),sd)
            age_col[i] = rand(object)
        elseif ismissing(data[i,:Age]) && sex== "male" && data[i,:Pclass]==3
            sd= std(male3)
            mm =mean(male3)
            object = Normal(mean(male3),sd)
            age_col[i] = rand(object)
        elseif ismissing(data[i,:Age]) && sex== "female" && data[i,:Pclass]==1
            sd= std(female1)
            mm = mean(female1)
            object = Normal(mean(female1),sd)
            age_col[i] = rand(object)
        elseif ismissing(data[i,:Age]) && sex== "female" && data[i,:Pclass]==2
            sd= std(female2)
            mm = mean(female2)
            object = Normal(mean(female2),sd)
            age_col[i] = rand(object)
        elseif ismissing(data[i,:Age]) && sex== "female" && data[i,:Pclass]==3
            sd= std(female3)
            mm =mean(female3)
            object = Normal(mean(female3),sd)
            age_col[i] = rand(object)
        else continue
        end
    end
return age_col
end

############################################################################

# DEAL WITH MISSING VALUES
# fare
function fare_missing_vals(fare_col)
    mm = mean(skipmissing(fare_col))
    sd = std(skipmissing(fare_col))
    object = Normal(mm,sd)
    fare_col[ismissing.(fare_col)] = rand(object,length(fare_col[ismissing.(fare_col)]))
    return fare_col
end

############################################################################

# DEAL WITH MISSING VALUES
# embarked
function embarked_missing_vals(embarked_col)
    mostfrequent= mode(embarked_col)
    for (i , s) in enumerate(embarked_col)
        if ismissing(s)
            embarked_col[i]= mostfrequent
        end
    end
    embarked_data = DataFrame()
    embarked_data.embarked = embarked_col
    ux = unique(embarked_data.embarked)
    data_new = transform(embarked_data, @. :embarked => ByRow(isequal(ux))=> Symbol(:embarked_,ux))
    return data_new[:,Not(1)]
end


############################################################################

# DEAL WITH MISSING VALUES
# cabin
function cabin_missing_vals(cabin_col)
    for i in  1:length(cabin_col)
        if ismissing(cabin_col[i])
            cabin_col[i] = "Undef"
        else
            cabin_col[i] = string(cabin_col[i][1])
        end
    end
    cabin_data = DataFrame()
    cabin_data.cabin = cabin_col
    ux = unique(cabin_data.cabin)
    data_new = transform(cabin_data, @. :cabin => ByRow(isequal(ux))=> Symbol(:cabin_,ux))
    return data_new[:,Not(1)]
end

############################################################################

# TRANFORM THE FEATURES
# sex onehot encoding
function sex_one_hot(sex_col)
    sex_data = DataFrame()
    sex_data.sex = sex_col
    ux = unique(sex_data.sex)
    data_new = transform(sex_data, @. :sex => ByRow(isequal(ux))=> Symbol(:sex_,ux))
    return data_new[:,Not(1)]
end

############################################################################

# TRANFORM THE FEATURES
# combine sibsp and parch
function sibsp_parch_combiner(sibsp,parch)
    relatives = sibsp .+ parch
    return relatives
end

############################################################################

# TRANFORM THE FEATURES
# extract title from name and one hot encode it
function title_extracter(name_col)
    for i in 1:length(name_col)
        d = split(name_col[i],".")[1]
        name_col[i] = split(d,", ")[2]
    end
    titles_data = DataFrame()
    titles_data.titles = name_col
    ux = unique(titles_data.titles)
    data_new = transform(titles_data, @. :titles => ByRow(isequal(ux))=> Symbol(:title_,ux))
    return data_new[:,Not(1)]
end

############################################################################

# TRANFORM THE FEATURES
# scale the numerical variables
function mean_normalization(col)
    norm = float.(col)
    for (i,s) in enumerate(col)
        norm[i]= (s - minimum(col)) / (maximum(col) - minimum(col))
    end
    return norm
end

############################################################################

# TRANFORM THE FEATURES
# one hot encode pclass
function pclass_one_hot(pclass_col)
    pclass_data = DataFrame()
    pclass_data.pclass = pclass_col
    ux = unique(pclass_data.pclass)
    data_new = transform(pclass_data, @. :pclass => ByRow(isequal(ux))=> Symbol(:pclass_,ux))
    return data_new[:,Not(1)]
end

############################################################################

# TRANFORM THE FEATURES
# extract ticket length
function ticket_number_length(ticket_col)
    ticketnumber=[]
    ticketlength=[]
    for i in ticket_col
        x= split(i, " ")
        if length(x)==1
            push!(ticketnumber, x[1])
        elseif length(x) == 2
            push!(ticketnumber, x[2])
        elseif length(x) == 3
            push!(ticketnumber, x[3])
        end
    end
    for i in ticketnumber
        push!(ticketlength, length(i))
    end
    return(ticketlength)
end

############################################################################