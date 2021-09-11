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
user_road_intersection= read_data("./data/user_roadintersection.csv")
user_corners = read_data("./data/user_corners.csv")
user_other = read_data("./data/user_other.csv")

# Available points
points_data = random_data_300 #* (100/30)
final_points_data = neighborhood_data #

# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
user_data = user_road
filename = "./data/out_images/testimage.png" #Final image for saving

#Create beta Values
beta_values = [points_data[i][4:6] for i in 1:length(points_data)]
final_beta_values = [final_points_data[i][4:6] for i in 1:length(final_points_data)]#TODO
choice_beta_values = [random_data[i][4:6] for i in 1:length(random_data)]

#Choose a user model
user_mode = user_expert
user_ideal = [0.001,0.95,0.05]

#Number of steps before making selection
guess_steps = 4

function _run(user_data,beta_values,final_beta_values,choice_beta_values)
    u_points = user_data            #Initial set of user points
    s_points = beta_values          #Points that can be suggested by algorithm
    f_points = final_beta_values    #Final points to be propagated
    o_points = choice_beta_values   #Points that the user can randomly select
    #Solver Definition
    randomMDP = FORollout(RandomSolver())
    solver = POMCPSolver(tree_queries=100, c=100.0, rng=MersenneTwister(1), tree_in_info=true,estimate_value = randomMDP)

    #Get statistics on initial set of user points
    avg_b = mean([u_points[a][4] for a in 1:length(u_points)])
    avg_r = mean([u_points[a][5] for a in 1:length(u_points)])
    avg_n = mean([u_points[a][6] for a in 1:length(u_points)])
    cov_b = std([u_points[a][4] for a in 1:length(u_points)])
    cov_r = std([u_points[a][5] for a in 1:length(u_points)])
    cov_n = std([u_points[a][6] for a in 1:length(u_points)])
    phi = [avg_b,avg_r,avg_n] 
    cov = [cov_b,cov_r,cov_n]
    phi = phi/norm(phi)  # Normalize
    cov = cov/norm(cov)
    #Any zero values must be made non-zero to make phi a positive vector in Dirichlet Dist.
    for a in 1:length(phi)
        if phi[a] == 0.0
            phi[a] = 0.001
        end
    end

    #Initilize Belief with Particle Filter
    #Create Gaussian Distribution
    p = 100 #Number of particles
    p_sample = 10 #Number of actions to consider --> Size of action space
    initial_belief = Dirichlet(phi) #Initialize belief. TODO: How to take variance into account?
    initial_p_set = [rand(initial_belief) for a in 1:p]
    p_belief = InjectionParticleFilter(initial_p_set,Int(round(p*0.05)),initial_belief)

    accepted_points = []
    user_points = []
    denied_points = []
    suggested_points = []

    best_points_idx,best_points_phi = find_similar_points(s_points,phi,p_sample,[])
    for step in 1:5
    # step = 1
        #Find the top 10 points for suggestion

        model_step = guess_steps+1-step  # Lets solver know how many steps are left
        PE_fun =  PE_POMDP(u_points,best_points_phi,o_points,f_points,user_mode,0.99,model_step)  # Define POMDP
        up = BootstrapFilter(PE_fun, 100)  # Unweighted particle filter
        planner = solve(solver, PE_fun)
        # history = collect(stepthrough(PE_fun, planner, up, "s,a,o,action_info", max_steps=guess_steps+1))
        a, info = action_info(planner, initialstate(PE_fun), tree_in_info=false)
        # inchrome(D3Tree(info[:tree], init_expand=3))
        
        # Action response 
        if a == "wait"
            #Randomly sample point based on user model
            new_user_point = sample_new_point(choice_beta_values,user_ideal,user_mode)
            #Update Particle Belief
            p_belief = update_PF(p_belief,PE_fun,a,new_user_point)
            push!(user_points,new_user_point) #Record keeping
        else
            #Find global point from suggested point
            suggested_idx = Int(best_points_idx[parse(Int64,a)])
            #Randomly sample user's response based on user model
            response = sample_user_response(s_points[suggested_idx],user_ideal,user_mode)
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
        #Sample from Particle filter
        particles = rand(p_belief.states,p_sample)
        #Replace sampled points
        for sample in 1:length(particles)
            idx,phi = find_similar_points(s_points,particles[sample],1,suggested_points)
            best_points_idx[sample] = idx[1]
            best_points_phi[sample] = phi[1]
        end
        
    end
    return p_belief,user_points,accepted_points,denied_points
end


belief,user_points,accepted_points,denied_points = _run(user_road,beta_values,final_beta_values,choice_beta_values)

#Visualization and image plotting
u_x,u_y = extract_xy(user_points,points_data)
a_x,a_y = extract_xy(accepted_points,points_data)
d_x,d_y = extract_xy(denied_points,points_data)
i_x = [user_data[i][1] for i in 1:length(user_data)]
i_y = [user_data[i][2] for i in 1:length(user_data)]

plot_image([i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename)