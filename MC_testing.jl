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

filename = "./data/out_images/testimage.png" #Final image for saving
filename_final = "./data/out_images/RoadEdge_final.png"
brier_filename = "./data/out_images/Brier_Score_Road.png" #Final image for saving
acc_filename = "./data/out_images/Accuracy_Road.png"
plot_title = "Accuracy for Building Case"
save_image = false # should images be saved
#Choose a user model
#user = user_expert
user_label = "Expert"
# If nn segmentation is required, calculate statistics in the following format
# Upper bounds could also be added by expanding feature n vector with max bound. Requires modification of user_dist in user_model.jl
    # user_ideal_nn = [[f1_mean,f1_std],[f2_mean,f2_std],....]
#Number of steps before making selection
# Other
#user_ideal_seg = [1e-5, 1e-5, 1.0]
#user_ideal_nn = [[0.10406552255153656, 0.0020979344844818115], [0.5734096050262452, 8.621215820312499e-05], [0.32167674899101256, 0.002654993534088135], [1e-05, 1e-05], [0.014076241850852966, 0.0006373345851898194], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [0.10211745351552963, 0.0031076580286026004], [1e-05, 1e-05], [1e-05, 1e-05], [0.4699282765388489, 0.001473522186279297], [0.1446827232837677, 0.0019847393035888673]]
#user_data = user_other
#user_type = "other"
#truth_data = [37, 38, 39, 40, 41]
#secondary_data = [6,7,8,9,10,22,23,24,25,26]

# Building
# user_ideal_seg = [0.876, 0.102006, 0.022004]
# user_ideal_seg = [0.99,0.01,0.01]
# user_ideal_nn = [[5.611148545410156, 3.6751314206677557], [40.754514786865236, 20.74506312483868], [38.24892883300781, 10.13161907748458], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [37.147728729248044, 8.140464585307537], [1e-05, 1e-05], [1.7446294280548096, 1.136748576368496], [35.64239311218262, 5.241401764333165], [12.73976674079895, 10.463944025736266], [1e-05, 1e-05], [1e-05, 1e-05], [39.2167423248291, 7.985361368713226], [55.322322082519534, 7.976847365299413]]
# user_data = user_building[1:2]
# user_type = "building"
# truth_data = [32, 33, 34, 35, 36]
# secondary_data = [22,23,24,25,26,1,2,3,4,5,16,17,18,19,20,21]

# Road
user_ideal_seg = [1e-05, 0.97, 0.030008]
user_ideal_seg = [1e-05,0.99,0.01]
user_ideal_nn = [[1e-05, 1e-05], [1e-05, 1e-05], [5.555718803405762, 0.2809410421740831], [1.0793159246444701, 0.2885174684541006], [1e-05, 1e-05], [0.38576561536979675, 0.20474108574479344], [1e-05, 1e-05], [2.032072591781616, 0.6365200213556083], [1e-05, 1e-05], [1e-05, 1e-05], [17.133995628356935, 4.820499085774453], [32.40111312866211, 2.1944726251520614], [1e-05, 1e-05], [1e-05, 1e-05], [6.844200801849365, 1.4105408165540374], [34.5301513671875, 0.17751821647618413]]
# user_ideal_nn = false
# user_data = user_road[1:2]
# user_data = vcat(user_road[1:2],random_data[5:7])
user_data = random_data[5:6]
user_type = "road"
truth_data = [27, 28, 29, 30, 31]
secondary_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Back
#user_ideal_seg = [0.372, 0.0020080000000000002, 0.6220000000000001]
#user_ideal_nn = [[4.318610281280518, 4.049102853140532], [25.268470191955565, 19.7067844866514], [25.479881858825685, 21.346860715033106], [1.73129124369812, 1.422232194099664], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [14.012442016601563, 6.057243077672148], [1e-05, 1e-05], [0.03397405551242828, 0.06792811102485656], [9.69793164730072, 6.776187880976638], [11.82856540136719, 9.7865366055892], [1e-05, 1e-05], [1e-05, 1e-05], [27.127032661437987, 19.916428327920308], [28.882941436767577, 19.014206493298715]]
#user_data = user_backdoor
#user_type = "backdoor"
#truth_data = [22,23,24,25,26]
#secondary_data = [1,2,3,4,5,32, 33, 34, 35, 36]

# Front
# user_ideal_seg = [0.54, 0.33199999999999996, 0.128006]
# user_ideal_nn = [[8.847843448120118, 12.851326194407378], [27.715496441986083, 38.197716820425676], [22.196852111816405, 12.32615385958663], [0.2192107655582428, 0.3885625852955098], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [22.557515716552736, 6.615757740995593], [1e-05, 1e-05], [0.48818081723022466, 0.9763416344604491], [33.26570091247559, 7.232214165523874], [19.296277568008424, 15.032755762533423], [1e-05, 1e-05], [1e-05, 1e-05], [32.65738162994385, 23.796163088253188], [37.97206153869629, 9.267686475204695]]
# user_data = user_frontdoor
# user_type = "frontdoor"
# truth_data = [1,2,3,4,5]
# secondary_data = [1,2,3,4,5,32, 33, 34, 35, 36]

# Corner
#user_ideal_seg = [0.298, 0.2060060, 0.502]
#user_ideal_nn = [[4.214598454101562, 8.429176908203125], [23.07781505041504, 22.21016878529063], [18.18980860710144, 14.785874159968039], [0.6877232442932129, 1.375426488586426], [1e-05, 1e-05], [1e-05, 1e-05], [1e-05, 1e-05], [16.424647045135497, 6.682789494377721], [1e-05, 1e-05], [0.42366331921386724, 0.8473066384277343], [22.659137153625487, 12.560100206768894], [12.249711794189455, 10.369474015431784], [1e-05, 1e-05], [1e-05, 1e-05], [21.529578685760498, 16.764819284370738], [30.49309959411621, 7.321425482659588]]
#user_data = user_corner
#user_type = "corner"
#truth_data = [16,17,18,19,20,21]
#secondary_data = [32, 33, 34, 35, 36,1,2,3,4,5,22,23,24,25,26]

# Road edges
# user_ideal_seg = [1e-05, 0.27599999999999997, 0.724]
# user_ideal_seg = [1e-05,0.5,0.5]
# user_ideal_nn = [[0.029287261827468873, 0.05855452365493774], [0.04013599859046936, 0.08025199718093873], [2.3334914749183655, 2.674239379582739], [4.955508420135498, 3.108996088683162], [1e-05, 1e-05], [2.342380830909729, 1.4732283557985926], [1e-05, 1e-05], [2.217990652519226, 3.913033456807154], [1e-05, 1e-05], [0.5940385260162354, 0.733404415789601], [13.246929794549942, 11.246110707197277], [10.749766969680786, 6.105504994570322], [1e-05, 1e-05], [0.09859951244163515, 0.19717902488327027], [2.3827168383178714, 3.2026035714198313], [12.619472789764405, 8.083957458953995]]
# # user_ideal_nn = false
# user_data = user_road_edges[1:3]
# user_type = "road_edge"
# truth_data = [6,7,8,9,10]
# secondary_data = [11,12,13,14,15,27, 28, 29, 30, 31]

# Road intersections
#user_ideal_seg = [1e-05, 0.558, 0.442]
#user_ideal_nn = [[1e-05, 1e-05], [1e-05, 1e-05], [3.9585901277918816, 4.406314752063278], [4.040073749687195, 3.3601555698640104], [1e-05, 1e-05], [1.898380154899597, 2.1228981830786333], [1e-05, 1e-05], [2.504726189903259, 2.821085549317644], [1e-05, 1e-05], [0.01095986769962311, 0.021899735399246216], [10.891311740875244, 5.356829320262097], [19.066756820678712, 7.816261426255367], [1e-05, 1e-05], [1e-05, 1e-05], [4.188327450088501, 4.842259682880136], [21.14072780609131, 7.742384523856947]]
#user_data = user_road_intersection
#user_type = "road_intersection"
#truth_data = [11,12,13,14,15]
# secondary_data = [11,12,13,14,15,27, 28, 29, 30, 31]

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

for u in 1:1
    println(u)
    brier_diff = Array{Float64}(undef,MC_runs,num_guess+1)
    global acc_score_hist = Array{Float64}(undef,MC_runs,num_guess)
    if u == 2
        local user = user_novice
        local user_label = "Novice POMDP"
    else
        local user = user_expert
        local user_label = "Expert POMDP"
    end
    for i in 1:MC_runs
        #Run Planner to find belief
        global belief,user_points,accepted_points,denied_points,hist =
            _run(user_data,user_ideal_seg,user_ideal_nn,points_data,final_points_data,random_data,user,num_guess)
        #Propagate belief onto new image
        local chosen = final_guess(final_points_data,belief,5)

        #Add chosen points to list
        for I in chosen
            vals = final_points_data[parse(Int64,I)][4:end]
            push!(chosen_set,vals)
            push!(chosen_idx,I)
        end
        push!(avg_belief,mean(belief.states))
        push!(std_belief,std(belief.states))
        
        #Parse and save belief data 
        for g in 1:length(hist.p_belief)
            brier_diff[i,g] = norm(mean(hist.p_belief[g].states)-user_ideal)
        end

        #Parse and save accuracy data
        r,c = size(hist.final_points)
        for g in 1:num_guess
            # Calculate accuracy at every guess point
            acc_score_hist[i,g] = accuracy_score(hist.final_points[:,g],truth_data,secondary_data)
            # print(hist.final_points[:,g])
        end

    end
    brier_plot = brier_crunch(brier_diff,MC_runs,num_guess)
    acc_plot = acc_crunch(acc_score_hist,MC_runs,num_guess)
    local x = range(1,num_guess,length = num_guess)
    if u == 1
        local x = range(0,num_guess+1,length = num_guess+1)
        global p_MSE = plot(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label)
        local x = range(1,num_guess,length = num_guess)
        global p_acc = plot(x,acc_plot[1,:],ribbon = acc_plot[2,:],label = user_label)
    else
        # plot!(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label)
        plot!(x,acc_plot[1,:],ribbon = acc_plot[2,:],label = user_label)
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
# user_avg_belief_exp = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
#Sum briers
user_brier_plot_exp = brier_crunch(user_avg_belief_exp,MC_runs,num_guess+length(user_set_betas))
#Plot user data
x = range(0,num_guess+1,length = num_guess+1)
#p = plot!(x,user_brier_plot_exp[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_exp[2,length(user_set_betas)+1:end],label =:"Expert Average")

#Initiate User Selection for NOVICE
user = user_novice
# user_avg_belief_nov = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
#Sum brier 
user_brier_plot_nov = brier_crunch(user_avg_belief_nov,MC_runs,num_guess+length(user_set_betas))
#Plot user data
x = range(0,num_guess+1,length = num_guess+1)
#p = plot!(x,user_brier_plot_nov[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_nov[2,length(user_set_betas)+1:end],label =:"Novice Average")

#Greedy Policy

title!(plot_title)
xlabel!("User Suggestions")
ylabel!("Accuracy")
# display(p)
if save_image
    # savefig(p_acc, brier_filename)
    savefig(p_acc, acc_filename)
end
#Plot all points on image
final_image = "./images/neighborhood_image.jpg"
p_neighborhood = plot_image(final_image,[],[],[p_x,p_y],[],filename_final,save_image)
#Initial Image extraction
u_x,u_y = extract_xy(user_points,points_data)
a_x,a_y = extract_xy(accepted_points,points_data)
d_x,d_y = extract_xy(denied_points,points_data)
i_x = [user_data[i][1] for i in 1:length(user_data)]
i_y = [user_data[i][2] for i in 1:length(user_data)]
guess_image = "./images/Image1_raw.png"
p_guess = plot_image(guess_image,[i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename,save_image)


# p_hist = plot(histogram([wrong_data,correct_data], bins=0:1:42), xticks=0:5:40)
# title!("Accuracy: "*string(length(correct_data)/(length(correct_data)+length(wrong_data))))
acc_score = accuracy_score(chosen_idx,truth_data,secondary_data)
cat_score = find_categories(chosen_idx)
names = ["Frontdoor","Edges","Intersections","Corners","Backdoor","Road","Building","Other"]
p_bar = bar(names,
            [cat_score[1],cat_score[2],cat_score[3],cat_score[4],cat_score[5],cat_score[6],cat_score[7],cat_score[8]],
            labels = names,
            title = "Accuracy: "*string(acc_score),
            ylabel = "Number of Guesses",
            legend = false)

l = @layout [a b; c d] 
full = plot(p_acc,p_bar,p_MSE,p_guess, layout = l)
display(full)

out_array = []
for i in user_ideal_nn
    push!(out_array, i[1])
end
# println(out_array)

mean_bl = mean(avg_belief)
println("Monte Carlo Testing Finished")
# print(mean_bl)
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
