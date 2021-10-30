
# MEAN SHIFT FUNCTIONS
############################################################################

# euclidian distance
using ML

############################################################################

# Kernel function
function kernel_function(distance,λ)
    if abs(distance) < λ
        weight = 1
    else
        weight = 0
    end
    return weight
end

############################################################################

# shift function
function shift_function(p,X,λ)
    weight = Float64[]
    for i in 1:size(X,1)
        distance = ML.euclid_distance(p,X[i,:])
        push!(weight,kernel_function(distance,λ))
    end
    reshape(weight,size(X,1),1)
    new_point = mean(weight .* X,dims=1)
    return new_point'
end

############################################################################

# mean shift function
function mean_shift_function(X,λ,treshold)
    centroids = []
    sample = deepcopy(X)
    for i in 1:size(X,1)
        validator = true
        while validator
            shifted_point = shift_function(sample[i,:],X,λ)
            shift_distance = ML.euclid_distance(shifted_point,sample[i,:])
            sample[i,:] = shifted_point
            if shift_distance < treshold
                validator = false
                if isempty(centroids)
                    push!(centroids,shifted_point)
                else
                    add = true
                    for p in centroids
                        dist = ML.euclid_distance(shifted_point,p)
                        if dist < treshold
                            add =  false
                            break
                        end
                    end
                    if add
                        push!(centroids,shifted_point)
                    end
                end
            end
        end
    end
    return centroids
end

############################################################################

# assign sample to closest cluster
function cluster_assign_mean_shift(data,k_matrix)
    distances = []
    clusters = []
    size_k = size(k_matrix,1)
    size_data = size(data,1)
    for data_point in 1:size_data
        for centroid in 1:size_k
            distance = ML.euclid_distance(data[data_point,:],k_matrix[centroid])
            push!(distances,distance)
        end
        index_of_min = argmin(distances)
        push!(clusters,index_of_min)
        distances = []
    end
    return clusters
end

############################################################################