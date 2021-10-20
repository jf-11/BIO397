
# IMAGE ROTATION
############################################################################
using Images
using ImageMagick
using FileIO
using Colors
############################################################################
img_path = "/Users/Joel/Desktop/BIO397/week02.1/flower.jpg"
img = load(img_path)
rot_matrix = [-1 1; 1 1]
rot_matrix2 = [1 -1; 1 1]
filename = "flower_rotated.jpg"
############################################################################
function img_rotater(img,rot_matrix,filename)
    x = size(img,1)
    y = size(img,2)
    height = x*2 + x÷2
    width = y*2 + y÷2
    color = RGB(0,0,0)
    image_rotated = fill(color,height,width)
    rot_matrix = rot_matrix
    center = rot_matrix * [x÷2,y÷2]
    scale = [abs(center[1] - height÷2),abs(center[2] - width÷2)]

    for row in 1:x
        for col in 1:y
            new_coord = rot_matrix * [row;col]
            image_rotated[new_coord[1]+scale[1],new_coord[2]+scale[2]] = img[row,col]
        end
    end
    save(filename,image_rotated)
end
############################################################################
img_rotater(img,rot_matrix,filename)
############################################################################
