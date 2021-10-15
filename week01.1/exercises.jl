# BIO397 Exercise set 1
# Joël Fehr

# 1. Create a vector of zeros of size 10 and set the second value to 1.

vector = zeros(Int64,10)
vector[2] = 1

# 2. Create a 2*3 matrix with values ranging from 1 to 6.

matrix = reshape(1:6,2,3)

# 3. Find the odd elements in vector [1,4,2,3,6,7].

vec = [1,4,2,3,6,7]
findall(a-> a%2!=0, vec)
vec[findall(a-> a%2!=0, vec)]

# 4. Multiply a 5*3 matrix of random numbers by a 3*2 matrix of random numbers.

rand(5,3) * rand(3,2)

# 5. Write a function that takes a string as its input and returns the string from backward.

function stringinverter(string)
    out = ""
    for i in 1:length(string)
        x = i - 1
        out = out * string[end-x]
    end
    return out
end

stringinverter("hello")

# 6. Write a function that checks whether a string is palindromic (the string is the same whether read from backward or forward).

function palindromechecker(string)
    string = lowercase(string)
    check = reverse(string)
    if check == string
        return "This is a palindrome"
    else
        return "This is not a palindrome"
    end
end

palindromechecker("Hello")
palindromechecker("ABBA")

# 7. Write a function that accepts a DNA sequence as a string and returns its RNA transcript.
# If the DNA has wrong letters, the function should complain.

function dnaconverter(dna)
    checklist = ['A','C','G','T']
    for i in dna
        if i ∉ checklist
            return "This is not a valid DNA sequence"
        end
    end
    rna = ""
    for i in dna
        if i == 'T'
            rna = rna * "U"
        else
            rna = rna * i
        end
    end
    return rna
end

dnaconverter("ACGTE")
dnaconverter("ACGTTACGTAGGTA")

# 8. Write a function that determines whether a word is an isogram (has no repeating letters, like the work “isogram”).

function isogram(string)
    string = lowercase(string)
    x = []
    for i in string
        if i ∉ x
            append!(x,i)
        else
            return "This string is not an isogram"
        end
    end
    return "This string is an isogram"
end

isogram("hello")
isogram("isogram")

# 9. Write a function that counts the number of elements of its input, whether the input is an array or a string.
# Then, it should return a new element that is of the same type as its input but with duplicate elements.

function duplicater(input)
    len = length(input)
    if typeof(input) == Vector{Int64}
        input = append!(input,input)
        return len, input
    else
        input = input*input
        return len, input 
    end
end

duplicater("Joel")
duplicater([1,2,3,4,5,6])

# 10. Write a function called nestedsum that takes an array of arrays of integers and adds up the elements from
# all of the nested arrays.

t = [[1, 2], [3], [4, 5, 6]]
function nestedsum(arrays)
    sum = 0
    for i in arrays
        for p in i
            sum = sum + p
        end
    end
    return sum
end

nestedsum(t)

# 11. Write a function that checks whether an array has duplicates.
# Use this function inside another function that returns the duplicated values and indices of an array, if they exist.

function duplicate(array)
    x = []
    for i in array
        if i ∉ x
            append!(x,i)
        else
            return true
        end
    end
    return false
end

one = [1,2,3,4,5,6]
two = [1,2,2,4,5,6,6]
duplicate(one)
duplicate(two)

function indexer(array)
    dupl = duplicate(array)
    x = []
    index = []
    if dupl == false
        return "No duplicate values."
    else
        for i in 1:length(array)
            if array[i] ∉ x
                append!(x,array[i])
            else
                append!(index,i)
            end
        end
    end
    return x, index
end

indexer(one)
indexer(two)

# 12. The geometry module:

mutable struct Point
    x::Float64
    y::Float64
end

x = Point(1,5)

mutable struct Circle2
    point::Point
    radius::Float64
end

circ = Circle2(x,2)

function area(circle::Circle2)
    return π * circle.radius^2
end

area(circ)

mutable struct Square
    side::Float64
end

sq = Square(4)

function area(square::Square)
    return square.side^2
end

area(circ)
area(sq)

function overlapping(circ1::Circle2,circ2::Circle2)
    distance = sqrt((circ1.point.x - circ2.point.x)^2 + (circ1.point.y - circ2.point.y)^2)
    if distance < circ1.radius + circ2.radius
        return "The circles do overlap each other."
    else
        return "The circles do not overlap each other."
    end
end

overlapping(circ,circ)
circc = Circle2(Point(5,5),1)
overlapping(circ,circc)

include("Geometry.jl")
x = Geometry.Point(1,2)
xx = Geometry.Point(3,3)
circ1 = Geometry.Circle2(x,5)
circ2 = Geometry.Circle2(xx,1)
Geometry.area(circ1)
Geometry.overlapping(circ1,circ2)

# 13. Use the following dataset, take a subset of it where the values of the first column are less than the mean of
# the fifth column. Sort the new data frame by the values of the first and the fifth columns. Write it to file.

using RDatasets
df = dataset("datasets","anscombe")

using Statistics
m = mean(df[!,5])
new_df = df[df[!, 1] .< m, :]
new_df = sort(new_df,:X1)

using CSV
filename = "myfile.txt"
CSV.write(filename,new_df)

# 14. Read the file you just wrote into a DataFrame. Check whether it is the same as the one you wrote.

f = open("myfile.txt")
for line in eachline(f)
    print(line)
end
close(f)

# 15. The dataset below is has the results of a national survey in Chile.
# You can read about it here. Column education is a categorical column.
# For each education category, create a new column in the dataset where the values are either 0 or 1, 1 pointing to
# that admission category. Then remove the education column. This process is called “one hot encoding” and as we will see
# is a central process in machine learning.

df = dataset("car","Chile")
using DataFrames
describe(df)
unique(df.Education)

ux = unique(df.Education);
new_df=transform(df,@. :Education => ByRow(isequal(ux)) .=> Symbol(:Education_, ux))
new_df = select!(new_df, Not(:Education))

# 16. Create scatter plots of age vs education and income vs education. Color the points according to sex.

using VegaLite

df |>
@vlplot(
    :point,
    x=:Age,
    y=:Education,
    color=:Sex,
    width=400,
    height=400
)

df |>
@vlplot(
    :point,
    x=:Income,
    y=:Education,
    color=:Sex,
    width=400,
    height=400
)

