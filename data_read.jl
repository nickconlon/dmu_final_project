#This file reads in the user supplied data and creates a data set that is used by the POMDP
using CSV

function read_data(file)
    # file = "random_data.csv"
    # random_data= Array{Float64,1}
    random_data = []

    #Read through the different rows
    for row in CSV.Rows(file)
        #Convert to Float64
        x = parse(Float64, row.x)
        y = parse(Float64,row.y)
        r = parse(Float64,row.r)
        building = parse(Float64,row.building)
        road = parse(Float64,row.road)
        none = parse(Float64,row.none)
        print(typeof([x,y,r,building,road,none]))
        push!(random_data,[x,y,r,building,road,none]) #Add to list
    end
    return random_data
end
# read_data("random_data.csv")
# println(random_data)

