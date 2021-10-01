#This script creates and runs Monte_Carlo testing 
using HypothesisTests

include("data_read.jl")
include("plot_image.jl")
include("user_model.jl")
include("PE_POMDP_Def.jl")
include("ParticleFilter_Def.jl")
include("Interaction.jl")

#Load point data
random_data = read_data("./data/random_data.csv")
random_data_300 = read_data("./data/random_data_300.csv")
neighborhood_data = read_data("./data/neighborhood_350.csv")

#User Data
user_frontdoor = read_data("./data/user_frontdoor.csv")
user_backdoor = read_data("./data/user_backdoor.csv")
user_building = read_data("./data/user_building.csv")
user_road = read_data("./data/user_road.csv")
test_points = read_data("./data/user_test.csv")
user_road_edges = read_data("./data/user_roadedges.csv")
points_data = random_data_300 #* (100/30)
final_points_data = neighborhood_data #

# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
user_data = user_road
filename = "./data/out_images/testimage.png" #Final image for saving
filename_final = "./data/out_images/test_final_image.png" #Final image for saving

#Choose a user model
user= user_expert
user_ideal = [0.001,0.90,0.05] #[%building,%road,%other]
user_dist = Dirichlet(user_ideal*user.certainty)

#Number of steps before making selection
num_guess = 5
MC_runs = 1000

chosen_set = []
chosen_idx = []
avg_belief = []
std_belief = []
for i in 1:MC_runs
    #Sample User Ideal
    user_sample_ideal = rand(user_dist,1)

    #Run Planner to find belief
    local belief,user_points,accepted_points,denied_points = _run(user_data,user_ideal,points_data,final_points_data,random_data,user,num_guess)
    #Propagate belief onto new image
    local chosen = final_guess(final_points_data,belief,num_guess)

    #Add chosen points to list
    for I in chosen
        vals = final_points_data[parse(Int64,I)][4:6]
        push!(chosen_set,vals)
        push!(chosen_idx,I)
    end
    push!(avg_belief,mean(belief.states))
    push!(std_belief,std(belief.states))
end

#Compare chosen points with sampled ideal
sample_avg_b = mean([chosen_set[a][1] for a in 1:length(chosen_set)])
sample_avg_r = mean([chosen_set[a][2] for a in 1:length(chosen_set)])
sample_avg_n = mean([chosen_set[a][3] for a in 1:length(chosen_set)])
sample_avg_set = [sample_avg_b,sample_avg_r,sample_avg_n]
println(sample_avg_b," ",sample_avg_r," ",sample_avg_n)


#Sample user dist
user_sample = rand(user_dist,MC_runs*num_guess)
user_b = mean([user_sample[:,a][1] for a in 1:length(user_sample[1,:])])
user_r = mean([user_sample[:,a][2] for a in 1:length(user_sample[1,:])])
user_n = mean([user_sample[:,a][3] for a in 1:length(user_sample[1,:])])
user_set = [user_b,user_r,user_n]
println(user_set)

initial_set =  [mean([user_data[a][4] for a in 1:length(user_data)]),mean([user_data[a][5] for a in 1:length(user_data)]),mean([user_data[a][6] for a in 1:length(user_data)])]
#Final values extraction
p_x,p_y = extract_xy(chosen_idx,final_points_data)

# guess_image = "./images/Image1_raw.png"
# final_image = "./images/neighborhood_image.jpg"
# plot_image(final_image,[],[],[p_y,p_x],[],filename_final)
mean_bl = mean(avg_belief)
std_bl = mean(std_belief)
ttest_b = OneSampleTTest(mean_bl[1],std_bl[1],MC_runs,user_ideal[1])
ttest_r = OneSampleTTest(mean_bl[2],std_bl[2],MC_runs,user_ideal[2])
ttest_n = OneSampleTTest(mean_bl[3],std_bl[3],MC_runs,user_ideal[3])