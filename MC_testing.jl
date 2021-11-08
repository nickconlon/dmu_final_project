#This script creates and runs Monte_Carlo testing 
using HypothesisTests
using Plots

include("data_read.jl")
include("plot_image.jl")
include("user_model.jl")
include("PE_POMDP_Def.jl")
include("ParticleFilter_Def.jl")
include("Interaction.jl")
include("stats_fun.jl")
include("all_data.jl")
#Load point data
#random_data = read_data("./data/random_data.csv")
#random_data_300 = read_data("./data/random_data_300.csv")
#neighborhood_data = read_data("./data/neighborhood_350.csv")

#User Data
#user_frontdoor = read_data("./data/user_frontdoor.csv")
#user_backdoor = read_data("./data/user_backdoor.csv")
#user_building = read_data("./data/user_building.csv")
#user_road = read_data("./data/user_road.csv")
#test_points = read_data("./data/user_test.csv")
#user_road_edges = read_data("./data/user_roadedges.csv")
#points_data = random_data_300 #* (100/30)
#final_points_data = neighborhood_data #

# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
user_data = user_other
filename = "./data/out_images/testimage.png" #Final image for saving
filename_final = "./data/out_images/RoadEdge_final.png"
brier_filename = "./data/out_images/Brier_Score_Road_Edge_1mSim.png" #Final image for saving
plot_title = "Mean Squared Error for Road Use Case"
save_image = false # should images be saved
#Choose a user model
user = user_expert
user_label = "Expert"
user_ideal_seg = [0.00000001, 0.0000001, 0.9]
user_ideal_nn = [[0.08346421148967743, 0.04177651971949443], [0.6093180656433106, 0.07183852277137336], [0.5235927999019623, 0.4045040188454945], [1e-05, 1e-05], [0.01119926002216339, 0.005628560484132032], [1e-05, 1e-05], [1e-05, 1e-05], [0.6368522535400392, 1.2736845070800782], [1e-05, 1e-05], [1e-05, 1e-05], [1.2064823041992185, 2.412944608398438], [0.46875049322843554, 0.7340491610814214], [1e-05, 1e-05], [1e-05, 1e-05], [0.618419075012207, 0.2973534003146158], [0.9566195964813232, 1.6243710679662033]]

# If nn segmentation is required, calculate statistics in the following format
# Upper bounds could also be added by expanding feature n vector with max bound. Requires modification of user_dist in user_model.jl
    # user_ideal_nn = [[f1_mean,f1_std],[f2_mean,f2_std],....]
#Number of steps before making selection
num_guess = 15
MC_runs = 100

chosen_set = []
chosen_idx = []
avg_belief = []
std_belief = []

#Consider nn feature vector case
if typeof(user_ideal_nn) == Bool
    user_ideal = user_ideal_seg
else
    user_ideal = vcat(user_ideal_seg,[a[1] for a in user_ideal_nn])
end

for u in 1:2
    brier_diff = Array{Float64}(undef,MC_runs,num_guess+1)
    if u == 2
        local user = user_novice
        local user_label = "Novice POMDP"
    else
        local user = user_expert
        local user_label = "Expert POMDP"
    end
    for i in 1:MC_runs
        #Run Planner to find belief
        local belief,user_points,accepted_points,denied_points,hist =
            _run(user_data,user_ideal_seg,user_ideal_nn,points_data,final_points_data,random_data,user,num_guess)
        #Propagate belief onto new image
        local chosen = final_guess(final_points_data,belief,num_guess)

        #Add chosen points to list
        for I in chosen
            vals = final_points_data[parse(Int64,I)][4:end]
            push!(chosen_set,vals)
            push!(chosen_idx,I)
        end
        push!(avg_belief,mean(belief.states))
        push!(std_belief,std(belief.states))
        
        #Parse and save belief data 
        for g in 1:length(hist)
            brier_diff[i,g] = norm(mean(hist[g].states)-user_ideal)
        end
    end
    brier_plot = brier_crunch(brier_diff,MC_runs,num_guess)
    local x = range(0,num_guess+1,length = num_guess+1)
    if u == 1
        global p = plot(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label)
    else
        plot!(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label)
    end
end

#user_set_mean = [mean([user_data[a][4] for a in 1:length(user_data)]),
#                mean([user_data[a][5] for a in 1:length(user_data)]),
#                mean([user_data[a][6] for a in 1:length(user_data)])]
user_set_betas = [user_data[i][4:end] for i in 1:length(user_data)]
#Final values extraction
p_x,p_y = extract_xy(chosen_idx,final_points_data)


#Initiate User Selection for EXPERT
user = user_expert
user_avg_belief_exp = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
#Sum brier scores
user_brier_plot_exp = brier_crunch(user_avg_belief_exp,MC_runs,num_guess+length(user_set_betas))
#Plot user data
x = range(0,num_guess+1,length = num_guess+1)
#p = plot!(x,user_brier_plot_exp[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_exp[2,length(user_set_betas)+1:end],label =:"Expert Average")

#Initiate User Selection for NOVICE
user = user_novice
user_avg_belief_nov = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
#Sum brier scores
user_brier_plot_nov = brier_crunch(user_avg_belief_nov,MC_runs,num_guess+length(user_set_betas))
#Plot user data
x = range(0,num_guess+1,length = num_guess+1)
#p = plot!(x,user_brier_plot_nov[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_nov[2,length(user_set_betas)+1:end],label =:"Novice Average")

#Greedy Policy

title!(plot_title)
xlabel!("Observation")
ylabel!("Mean Squared Error")
display(p)
if save_image
    savefig(p, brier_filename)
end
#Plot all points
guess_image = "./images/Image1_raw.png"
final_image = "./images/neighborhood_image.jpg"
plot_image(final_image,[],[],[p_y,p_x],[],filename_final,save_image)

# #Finding means
# #Compare chosen points with sampled ideal
# sample_avg_b = mean([chosen_set[a][1] for a in 1:length(chosen_set)])
# sample_avg_r = mean([chosen_set[a][2] for a in 1:length(chosen_set)])
# sample_avg_n = mean([chosen_set[a][3] for a in 1:length(chosen_set)])
# sample_avg_set = [sample_avg_b,sample_avg_r,sample_avg_n]
# # println(sample_avg_b," ",sample_avg_r," ",sample_avg_n)


# #Sample user dist
# user_sample = rand(user_dist,MC_runs*num_guess)
# user_b = mean([user_sample[:,a][1] for a in 1:length(user_sample[1,:])])
# user_r = mean([user_sample[:,a][2] for a in 1:length(user_sample[1,:])])
# user_n = mean([user_sample[:,a][3] for a in 1:length(user_sample[1,:])])
# user_set = [user_b,user_r,user_n]
# # println(user_set)

# #TTesting
# mean_bl = mean(avg_belief)
# std_bl = mean(std_belief)
# ttest_b = OneSampleTTest(mean_bl[1],std_bl[1],MC_runs,user_ideal[1])
# ttest_r = OneSampleTTest(mean_bl[2],std_bl[2],MC_runs,user_ideal[2])
# ttest_n = OneSampleTTest(mean_bl[3],std_bl[3],MC_runs,user_ideal[3])
