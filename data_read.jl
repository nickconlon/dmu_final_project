#This file reads in the user supplied data and creates a data set that is used by the POMDP
using CSV

function read_data(file)
    # file = "random_data.csv"
    # random_data= Array{Float64,1}
    random_data = []
    ffile = CSV.File(file)
    columns = length(ffile.names)
    #Read through the different rows
    for row in ffile
        #Convert to Float64
        x = row[1]
        y = row[2]
        r = row[3]
        features = Array{Float64}([x,y,r])
        for feature in 4:columns
            push!(features, row[feature])
        end
        push!(random_data,features) #Add to list
    end
    return random_data
end
# read_data("random_data.csv")
# println(random_data)

