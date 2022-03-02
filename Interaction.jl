#Interaction Script
# This script iterates through possible points to suggest. 
# Input:
    #User data
    #Random data for suggestions
    #Data to apply suggestions on


#Include other functions

include("data_read.jl")
include("plot_image.jl")
include("user_model.jl")
include("PE_POMDP_Def.jl")
include("ParticleFilter_Def.jl")
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
# user_data = user_frontdoor
filename = "./data/out_images/testimage.png" #Final image for saving
filename_final = "./data/out_images/test_final_image.png" #Final image for saving

#Choose a user model
user = user_expert
user_ideal_seg = [0.00000001, 0.0000001, 0.9]
user_ideal_nn = [[0.08346421148967743, 0.04177651971949443], [0.6093180656433106, 0.07183852277137336], [0.5235927999019623, 0.4045040188454945], [1e-05, 1e-05], [0.01119926002216339, 0.005628560484132032], [1e-05, 1e-05], [1e-05, 1e-05], [0.6368522535400392, 1.2736845070800782], [1e-05, 1e-05], [1e-05, 1e-05], [1.2064823041992185, 2.412944608398438], [0.46875049322843554, 0.7340491610814214], [1e-05, 1e-05], [1e-05, 1e-05], [0.618419075012207, 0.2973534003146158], [0.9566195964813232, 1.6243710679662033]]

user_data = user_other
#Number of steps before making selection
num_guess = 1

struct history
    p_belief::Vector{InjectionParticleFilter}
    final_points::Matrix{String}
end

function _run(user_data,user_ideal_seg,user_ideal_nn,guess_points,final_points,choice_points,user_mode,guess_steps)
    #Input:
    #   user_data = [p_x,p_y,radius,%building,%road,%other] Full data vector
    #   user_ideal = [%building,%road,%other] Desired feature vector
    #   guess_points = [p_x,p_y,radius,%building,%road,%other] Full data vector
    #   final_points = [p_x,p_y,radius,%building,%road,%other] Full data vector
    #   choice_points = [p_x,p_y,radius,%building,%road,%other] Full data vector
    #Output:
    #   p_belief = Particle Filter object --> See ParticleFilter_Def.jl
    #   
    #   Following outputs are all in consistent format: ["idx1","idx2",...]
    #       user_points: Set of points chosen by the user as part of wait action
    #       accepted_points: Set of points suggested and accepted by user
    #       denied_points: Set of points suggested and denied by user
    ####

    #Extract beta Values
    beta_values = [guess_points[i][4:end] for i in 1:length(guess_points)]
    final_beta_values = [final_points[i][4:end] for i in 1:length(final_points)]
    choice_beta_values = [choice_points[i][4:end] for i in 1:length(choice_points)]

    u_points = user_data            #Initial set of user points
    s_points = beta_values          #Points that can be suggested by algorithm
    f_points = final_beta_values    #Final points to be propagated
    o_points = choice_beta_values   #Points that the user can randomly select
    #Solver Definition
    randomMDP = FORollout(RandomSolver())
    solver = POMCPSolver(tree_queries=100, c=5.0, rng=MersenneTwister(1), tree_in_info=true,estimate_value = randomMDP)

    #Get statistics on initial set of user points
    phi = Array{Float64}(undef,length(u_points[1])-3)
    cov = Array{Float64}(undef,length(u_points[1])-3)
    data_vec = Array{Array{Float64}}(undef,length(u_points[1])-3)
    for i in 4:length(u_points[1])
        ave_i = mean([u_points[a][i] for a in 1:length(u_points)])
        phi[i-3] = ave_i
        cov_i = std([u_points[a][i] for a in 1:length(u_points)])
        cov[i-3] = cov_i
        data_vec[i-3] = [ave_i,cov_i]
    end

    phi[1:3] = phi[1:3]/norm(phi[1:3])  # Normalize
    # cov = cov/norm(cov)
    #Any zero values must be made non-zero to make phi a positive vector in Dirichlet Dist.
    for a in 1:length(phi)
        if phi[a] == 0.0
            phi[a] = 1e-5
        end
    end

    #Initilize Belief with Particle Filter
    #Create Gaussian Distribution
    p = 1000 #Number of particles
    p_sample = 10 #Number of user actions to consider --> Size of action space
    greedy = false # Greedy policy toggle
    # println(phi)
    if typeof(user_ideal_nn) == Bool # If only using seg
        p_belief = init_PF(phi[1:3],false,p)
    elseif typeof(user_ideal_seg) == Bool # If only using nn 
        p_belief = init_PF(false,data_vec[4:end],p)
    else
        p_belief = init_PF(phi[1:3],data_vec[4:end],p)
    end
    accepted_points = []
    user_points = []
    denied_points = []
    suggested_points = []
    
    p_belief_history = Array{InjectionParticleFilter}(undef,guess_steps+1)
    guess_points = Array{String}(undef,5,guess_steps+1)
    hist = history(p_belief_history,guess_points)
    best_points_idx,best_points_phi = find_similar_points(s_points,phi,p_sample,[])
    for step in 1:guess_steps+1
        #Save particle belief
        hist.p_belief[step] = p_belief
        #Initialize POMDP with new action space: Figure out best action
        #   Input into POMDP is only the beta values
        #   Output is the index of the suggested value or "wait"
        model_step = guess_steps+1-step  # Lets solver know how many steps are left
        PE_fun =  PE_POMDP(u_points,best_points_phi,o_points,f_points,user_mode,0.99,model_step)  # Define POMDP
        planner = solve(solver, PE_fun)
        # Solver option
        if greedy == true # Greedy policy
            best_points = [s_points[parse(Int64,idx)] for idx in best_points_idx]
            a_idx,a_phi = find_similar_points(best_points,mean(p_belief.states),1,[])
            a = a_idx[1]
        else # POMCP solver
            best_phi = mean(p_belief.states)
            mid_state(rng) = State(best_phi,[],model_step)
            a, info = action_info(planner, ImplicitDistribution(mid_state), tree_in_info=false)
            # inchrome(D3Tree(info[:tree], init_expand=3))
        end
        # Action response 
        if a == "wait"
            #Randomly sample point based on user model
            new_user_idx,new_user_point = sample_new_point(choice_beta_values,user_ideal_seg,user_ideal_nn,user_mode,user_points)
            #Update Particle Belief
            p_belief = update_PF(p_belief,PE_fun,a,new_user_point)
            push!(user_points,new_user_idx[1]) #Record keeping
        else
            #Find global point from suggested point
            suggested_idx = best_points_idx[parse(Int64,a)]
            #Randomly sample user's response based on user model
            response = sample_user_response(s_points[parse(Int64,suggested_idx)],user_ideal_seg,user_ideal_nn,user_mode)
            #Update Particle Belief
            p_belief = update_PF(p_belief,PE_fun,a,response)
            
            #Record Keeping
            if response=="accept"
                push!(accepted_points,string(suggested_idx))
            else
                push!(denied_points,string(suggested_idx))
            end
            push!(suggested_points,string(suggested_idx))
        end

        #Update set of points to iterate through
        # #Sample from Particle filter
        # particles = mean(p_belief.states)
        # #Replace sampled points
        # for sample in 1:p_sample
        #     idx,phi = find_similar_points(s_points,particles,1,suggested_points)
        #     best_points_idx[sample] = idx[1]
        #     best_points_phi[sample] = phi[1]
        # end
        # Semi random sampling
        particles = rand(p_belief.states,p_sample)
        #Replace sampled points
        for sample in 1:length(particles)
            idx,phi = find_similar_points(s_points,particles[sample],1,suggested_points)
            best_points_idx[sample] = idx[1]
            best_points_phi[sample] = phi[1]
        end

        # Propagate set of final points for record keeping
        chosen = final_guess(final_points,p_belief,5)
        hist.final_points[:,step] = chosen
    end
    return p_belief,user_points,accepted_points,denied_points,hist
end

belief,user_points,accepted_points,denied_points = _run(user_data,user_ideal,false,points_data,final_points_data,random_data,user,num_guess)

# #Propagate belief onto new image
#  chosen = final_guess(final_points_data,belief,10)


# #Visualization and image plotting
# #Initial Image extraction
# u_x,u_y = extract_xy(user_points,points_data)
# a_x,a_y = extract_xy(accepted_points,points_data)
# d_x,d_y = extract_xy(denied_points,points_data)
# i_x = [user_data[i][1] for i in 1:length(user_data)]
# i_y = [user_data[i][2] for i in 1:length(user_data)]

# #Final values extraction
# p_x,p_y = extract_xy(chosen,final_points_data)

# guess_image = "./images/Image1_raw.png"
# final_image = "./images/neighborhood_image.jpg"
# plot_image(guess_image,[i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename)
# plot_image(final_image,[],[],[p_y,p_x],[],filename_final)