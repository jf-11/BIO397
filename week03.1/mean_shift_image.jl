
# MEAN SHIFT IMAGE EXERCISE
############################################################################

using Images
using FileIO
using ImageMagick

############################################################################

# load the Image
image_path = "/Users/Joel/Desktop/BIO397/week03.1/landscape.png"
img = load(image_path)

############################################################################

R = float(red.(img))
G = float(green.(img))
B = float(blue.(img))

Y = 0.299*R + 0.587*G + 0.114*B
U = -0.147*R - 0.289*G + 0.436*B
V = 0.615*R - 0.515*G - 0.100*B

############################################################################

# determine the coordinates of the points
x = []
y = []
for xx in 1:height(img)
    for yy in 1:width(img)
        push!(x, xx)
        push!(y, yy)
    end
end

coords = hcat(x,y)

############################################################################

# put it together to a feature matrix
Y = reshape(Y, size(coords, 1), 1)
U = reshape(U, size(coords, 1), 1)
V = reshape(V, size(coords, 1), 1)

feature_matrix = hcat(coords,Y,U,V)

# now we have to scale the matrix
using Statistics
feature_matrix_normalized = zeros(size(feature_matrix))
for i in 1:size(feature_matrix,2)
    mm = mean(feature_matrix[:,i])
    sd = std(feature_matrix[:,i])
    for p in 1:size(feature_matrix,1)
        feature_matrix_normalized[p,i] = (feature_matrix[p,i] - mm) / (maximum(feature_matrix[:,i])-minimum(feature_matrix[:,i]))
    end
end

############################################################################

# now we can use ScikitLearn
using ScikitLearn
@sk_import cluster: MeanShift

model = MeanShift(bandwidth=0.5)
fitted = model.fit(feature_matrix_normalized)

centroids = model.cluster_centers_
labels = model.labels_

############################################################################

coordinates = centroids[:,1:2]
YUV = centroids[:,3:end]
R = YUV[:,1] .+ 1.140 .* YUV[:,3]
G = YUV[:,1] .- 0.395 .* YUV[:,2] .- 0.581*YUV[:,3]
B = YUV[:,1] .+ 2.032 .* YUV[:,3]

############################################################################

unique(labels)

# the labels start with 0 so we have to make labels + 1, otherwise we have
# a bounds error...
labels = labels .+ 1

C = N0f8.(collect(0:1/length(R):1))

segmented_image = img

for i in 1:length(img)
    segmented_image[i] = RGB(C[labels[i]],C[labels[i]],C[labels[i]])
end

save("segmented_image.png", segmented_image)

############################################################################

