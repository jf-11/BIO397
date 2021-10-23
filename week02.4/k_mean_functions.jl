
# K-MEAN FUNCTIONS
############################################################################

# calculate euclidian distance
euclid_distance(p, q) = sqrt(sum((p .- q).^2))

############################################################################

# assign sample to closest cluster
function cluster_assign(data,k_matrix)
    distances = []
    clusters = []
    size_k = size(k_matrix,1)
    size_data = size(data,1)
    for data_point in 1:size_data
        for centroid in 1:size_k
            distance = euclid_distance(reshape(data[data_point,:],1,2),reshape(k_matrix[centroid,:],1,2))
            push!(distances,distance)
        end
        index_of_min = argmin(distances)
        push!(clusters,hcat(reshape(data[data_point,:],1,2),index_of_min))
        distances = []
    end
    return clusters
end

############################################################################

# create a function that formats the output
function formatter(cluster_object)
    x = []
    y = []
    cluster = []
    for i in cluster_object
        push!(x,i[1])
        push!(y,i[2])
        push!(cluster,i[3])
    end
    data = hcat(x,y,cluster)
    return data
end

############################################################################

# calculate the mean of the centroid
function calculate_mean(formatted_matrix,k_matrix)
    bools = []
    meansx = []
    meansy = []
    for i in 1:size(k_matrix,1)
        push!(bools,formatted_matrix[:,3].==i)
    end
    for i in 1:size(k_matrix,1)
        push!(meansx,mean(formatted_matrix[bools[i],1]))
        push!(meansy,mean(formatted_matrix[bools[i],2]))
    end
    means_object = hcat(meansx,meansy)
    return means_object
end

############################################################################

# function to assign means to centroids
function means_to_cluster(means_object,k_matrix)
    k_matrix = means_object
    return k_matrix
end

############################################################################

# FINAL FUNCTION

function k_means_clustering(data,k)
    shuff = shuffle(1:size(data,1))
    x = data[shuff[1:k],1]
    y = data[shuff[1:k],2]
    k_matrix = hcat(x,y)
    diff = 100
    formatted_cluster = randn(size(data,1),k)
    while diff > 0.01
        initial_cluster = cluster_assign(data,k_matrix)
        formatted_cluster = formatter(initial_cluster)
        calculated_means = calculate_mean(formatted_cluster,k_matrix)
        diff = abs(sum(calculated_means .- k_matrix))
        k_matrix = calculated_means
    end
    return k_matrix,formatted_cluster
end


############################################################################