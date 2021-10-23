
# KNN FUNTIONS
############################################################################

function get_vowels(name)
    sum_vowels = 0
    ss = lowercase(name)
    vowels= ['a', 'e', 'i', 'o', 'u' ]
    for i in name
        if i in vowels
            sum_vowels= sum_vowels+1
        end
    end
    return sum_vowels
end

############################################################################

function get_hard_consonant(name)
    ss = lowercase(name)
    consonant= ['b', 'd', 'g', 'p', 't', 'k' ]
    sum_hard_cons = 0
    for i in name
        if i in consonant
            sum_hard_cons = sum_hard_cons+1
        end
    end
    return sum_hard_cons
end

############################################################################

function get_length(name)
    name_length = length(name)
    return name_length
end

############################################################################

function get_first_letter(name)
    alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in 1: length(alph)
        if alph[i] == name[1]
            return i
        end
    end
end

############################################################################

function get_last_letter(name)
    alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in 1: length(alph)
        if alph[i] == name[length(name)]
            return i
        end
    end
end

############################################################################

function get_all_features(name)
    all_features = []
    push!(all_features, get_vowels(name),get_hard_consonant(name),get_length(name),get_first_letter(name), get_last_letter(name))
    return all_features
end

############################################################################

function calc_euclid_dist(input_data, name_to_test)
     data_features = get_all_features.(input_data[:,:Column1])
     name_features = get_all_features(name_to_test)
     distance = []
     for i in data_features
        distance_i = sqrt(sum((i .- name_features).^2))
        push!(distance, distance_i)
     end
     return distance
end

############################################################################

#now we want to sort these results
function predict_gender(input_data, name_to_test,k)
    euclid_distt = calc_euclid_dist(input_data, name_to_test)
    index_sorted = sortperm(euclid_distt)
    temporary = input_data[index_sorted[1:k],:Column2]
    return mode(temporary)
end

############################################################################