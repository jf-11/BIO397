
# ANOMALY DETECTION FUNCTIONS
############################################################################

function outlier_detecter(X,σ)
    outliers = []
    outliers_encoding = []
    for sample in 1:size(X,1)
        probab = []
        for feature in 1:size(X,2)
            μ = mean(X[:,feature])
            sd = std(X[:,feature])
            normal = Normal(μ,sd)
            prob = pdf(normal,X[sample,feature])
            push!(probab,prob)
        end
        if prod(probab) < σ
            push!(outliers,sample)
            push!(outliers_encoding,1)
        else
            push!(outliers_encoding,0)
        end
    end
    return outliers,outliers_encoding
end

############################################################################
