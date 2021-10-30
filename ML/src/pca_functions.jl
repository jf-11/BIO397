
# PCA FUNCTIONS
############################################################################

# cov matrix, eigenvectors
function pca_function(X,k)
    covariance_matrix = cov(X)
    eigenvalues = eigvals(covariance_matrix)
    relevant = sortperm(eigenvalues,rev=true)[1:k]
    eigenvectors = eigvecs(covariance_matrix)
    feature_vector = eigenvectors[:,relevant]
    explained_variances = eigenvalues ./ sum(eigenvalues)
    total_exp_variance = sum(explained_variances[relevant])
    final_data = feature_vector' * X'
    return final_data',total_exp_variance
end

############################################################################
