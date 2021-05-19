using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, QMDP
using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPModelTools
using D3Trees
using Random
using ParticleFilters
using POMDPPolicies: FunctionPolicy, alphavectors
using Plots
using SARSOP: SARSOPSolver
using POMDPs, POMDPModels, POMDPSimulators, BasicPOMCP
using LinearAlgebra
using Distances
using Statistics

#Include other functions
include("data_read.jl")
include("plot_image.jl")

#Load point data
random_data = read_data("data/random_data.csv")
random_data_300 = read_data("data/random_data_300.csv")
user_frontdoor = read_data("data/user_frontdoor.csv")
user_backdoor = read_data("data/user_backdoor.csv")
user_building = read_data("data/user_building.csv")
user_road = read_data("data/user_road.csv")
#test_points = read_data("data/user_test.csv")
user_road_edges = read_data("data/user_roadedges.csv")
user_road_intersection= read_data("data/user_roadintersection.csv")
user_corners = read_data("data/user_corners.csv")
user_other = read_data("data/user_other.csv")

# Available points
points_data = random_data_300 #* (100/30)
# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
user_data = user_corners
filename = "data/out_images/corners3.png"

#Create beta Values
beta_values = [points_data[i][4:6] for i in 1:length(points_data)]

# Function returns the feature values for a given point
function beta(s, a)    
    # zero rewards for waiting
    if a == "wait"
        return [0,0,0]
    else
        #Extract point distribution
        a_num = parse(Int64,a)
        vec = beta_values[a_num] 
     end
end


function create_state()
    # This function creates an initial Gaussian state distribution. The mean is initialized as the mean of the user
    # selected points and the covariance is the variance of the user points
    POI = user_data
    #Take mean of observed points and add noise
    avg_b = mean([POI[a][4] for a in 1:length(POI)])*2
    avg_r = mean([POI[a][5] for a in 1:length(POI)])
    avg_n = mean([POI[a][6] for a in 1:length(POI)])
    cov_b = std([POI[a][4] for a in 1:length(POI)])
    cov_r = std([POI[a][5] for a in 1:length(POI)])
    cov_n = std([POI[a][6] for a in 1:length(POI)])
    vec = [avg_b,avg_r,avg_n]
    vec = vec/norm(vec)
    return vec,cov_b, cov_r,cov_n
end

# Initialize state
global_phi,cov_b,cov_r,cov_n = create_state()
global_cov = [cov_b^2 0 0; 0 cov_r^2 0; 0 0 cov_n^2]
T = 10  # Steps before final guess is made

function sample_initial_state(rng)
    # Determine the starting state. Calls the global state variable and puts it in a Vector
    # Can be modified to add noise
    avg_b = global_phi[1]
    avg_r = global_phi[2]
    avg_n = global_phi[3]
    phi = [avg_b, avg_r, avg_n]
    # phi = phi/norm(phi) # Normalize
   
    init_cov = global_cov
    return State(phi,init_cov,0)
end

function KalmanUpdate(b,o)
    # Basic Kalman update with no transition dynamics
    #Assume that all observations update all classes
    o_var = 0.15
    cov_o = [o_var^2 0 0; 0 o_var^2 0; 0 0 o_var^2]

    #Time Update
    mu_p = b.phi  # Increment the mean
    cov_p = b.cov # Increment the covariance
    K = cov_p*I*inv(cov_p+cov_o)  #Kalman Gain

    #Measurement Update
    mu_b = mu_p + K*(o-mu_p)
    cov_b = (I-K)*cov_p

    return State([mu_b[1],mu_b[2],mu_b[3]],cov_b,b.step)
end

function make_observations()
    # Creates a list of potential observations in an array of strings
    total_act = length(points_data)
    a = Array{String}(undef, total_act+3)
    for i in 1:total_act
        a[i] = string(i)
    end
    a[end-1] = "accept"
    a[end] = "deny"
    a[end-2] = "no response"
    return a
end

function make_actions()
    # Creates a list of potential actions in an array of strings
    total_act = length(points_data)
    a = Array{String}(undef, total_act+1)
    for i in 1:total_act
        a[i] = string(i)
    end
    a[end] = "wait"
    return a
    # deleteat!(a, findall(x->x=="1", a))
end

function similarity(x, y)
    # Similarity metric calculator
    return 1-cosine_dist(x, y)
end

struct State
    phi::Vector{Float64}
    cov::Array{Float64,2}    
    step::Int64
end

actions_list = make_actions()
observations_list = make_observations()
# initial_state = sample_initial_state(MersenneTwister(1))

m = QuickPOMDP(
    actions = actions_list,
    # Attempt at creating dynamic action space. Doesn't work in current version
    # function()
    #     if s.step<T
    #         return actions_list
    #     else
    #         return make_action()  # Choose the single best point that most closely approximates user's distribution
    #     end
    # end,

    observations = observations_list,
    initialstate = ImplicitDistribution(sample_initial_state), # hidden state distribution
    discount = 0.95,

    transition = function (s, a)
        s.step += 1  # Increment step counter
        return Deterministic(s) # The state never changes
    end,

    observation = function (s, a, sp)
        if a == "wait"
            # points already accepted should get zero probability mass
            # [p1, p2,... pn] = (p(p1), p(p2)) = normalized(sim(p1, s.phi), sim(p2, s.phi), ...)
            p_a = []
            acts = []
            for (i, act) in enumerate(actions(m))
                if act != "wait"
                    b = beta_values[parse(Int64,act)]#(x,y,z)
                    out = create_state()
                    sim = similarity(out[1], b)
                    if sim < 0.5
                        sim = 0.0001
                    else
                        sim = sim*sim
                    end
                    push!(p_a,sim)
                    push!(acts, act)
                end
            end
            p_a = length(p_a)>0 ? p_a/norm(p_a) : p_a  #Catch cases where A is empty
            #available_actions = actions(m)[1:end-1]
            return SparseCat(acts, p_a)
            # distribution over all operator add points
        else
            # agent is suggesting 'a'
            # operator likes s.phi
            # points already accepted should get denied
            sim = similarity(s.phi, beta_values[parse(Int64,a)])
            sim = sim*0.8
            # sim = sim/norm(sim)

            p = [sim, 1-sim]
            return SparseCat(["accept", "deny"], p)
        end
    end,

    reward = function (s, a)
        r = 0.0
        if s.step < T
            if a == "wait"
                r = 0.0  #Small negative for suggesting?
            else 
                r = -1.0
            end            
        else
            vec = beta_values[parse(Int64,a)] 
            r = similarity(s.phi, vec)
        end
        return r
    end,

    isterminal = s->s.step == T+1,
)

pomdp =  m  # Define POMDP
up = BootstrapFilter(pomdp, 1000)  # Unweighted particle filter
solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
planner = solve(solver, pomdp)

#Step through simulated process
#actions_test
#println(observations(pomdp))
#deleteat!(actions_list, findall(x->x=="3", actions_list))
#deleteat!(observations_list, findall(x->x=="3", observations_list))
#println(observations(pomdp))
#vec = beta_values[3]
#deleteat!(beta_values, findall(x->x==vec, beta_values))


function _run()
    # Function runs through total interaction. 
    suggestions_allowed = 4
    accepted_points = []
    user_points = []
    denied_points = []

    suggestions_received = [0]

    #for suggestions in 1:4
    while suggestions_received[1] < suggestions_allowed
        # update the state.phi values on before re initializing the POMCP
        #global_phi[1] = 0
        #global_phi[3] = 1
        pomdp =  m  # Define POMDP
        up = BootstrapFilter(pomdp, 1000)  # Unweighted particle filter
        solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
        planner = solve(solver, pomdp)
        println("----STEP----")
        for (s, a, o, ai) in stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3)
            # println(s.phi)
            println("State was $s,")
            println("action $a was taken,")
            println("and observation $o was received.\n")
        
            # remove o if o is a point
            # remove a if o is accept
            # println(observations_list)
            # println(actions_list)
            if o == "accept"
                #remove a from actions & observations
                deleteat!(actions_list, findall(x->x==a, actions_list))
                deleteat!(observations_list, findall(x->x==a, observations_list))
                
                #Increment suggestions
                suggestions_received[1]+=1

                new_state = KalmanUpdate(s,beta_values[parse(Int64,a)]) #Update estimate of mean and covariance
                # Update state to account for observation
                global_phi[1] = new_state.phi[1]
                global_phi[2] = new_state.phi[2]
                global_phi[3] = new_state.phi[3]
                global_cov[1:3] = new_state.cov[1:3]
                global_cov[4:6] = new_state.cov[4:6]
                global_cov[7:9] = new_state.cov[7:9]

                # Save off the point
                push!(accepted_points, a)
                break
            end

            if o == "deny"
                #If the user denies a suggestion, kick out potential point from action and observation space 
                deleteat!(actions_list, findall(x->x==a, actions_list))
                deleteat!(observations_list, findall(x->x==a, observations_list))

                #Update coveriance to increase uncertainty
                # new_cov= global_cov*1.2
                # new_cov = global_cov/similarity(a,global_phi)
                # inv_beta = ones(3) - beta_values[parse(Int64,a)]
                # new_state = KalmanUpdate(s,inv_beta)
                # global_phi[1] = new_state.phi[1]
                # global_phi[2] = new_state.phi[2]
                # global_phi[3] = new_state.phi[3]
                # global_cov[1:3] = new_state.cov[1:3]
                # global_cov[4:6] = new_state.cov[4:6]
                # global_cov[7:9] = new_state.cov[7:9]
                # println(global_cov)
                # Save off the point
                push!(denied_points, a)
                break
            end
            #If the user selects a point 
            if o != "accept" || o != "deny" || o != "no response"
                # remove o from actions & observations
                deleteat!(actions_list, findall(x->x==o, actions_list))
                deleteat!(observations_list, findall(x->x==o, observations_list))

                new_state = KalmanUpdate(s,beta_values[parse(Int64,o)]) #Update estimate of mean and covariance
                # Update state to account for observation
                global_phi[1] = new_state.phi[1]
                global_phi[2] = new_state.phi[2]
                global_phi[3] = new_state.phi[3]
                global_cov[1:3] = new_state.cov[1:3]
                global_cov[4:6] = new_state.cov[4:6]
                global_cov[7:9] = new_state.cov[7:9]

                # Save off the point
                push!(user_points, o)
            end
            if length(observations_list) <= 3 || length(actions_list) <= 1 || suggestions_received[1] >= suggestions_allowed
                println("out of suggestions")
                break
            end
        end
        # u_x,u_y = extract_xy(user_points,points_data)
        # a_x,a_y = extract_xy(accepted_points,points_data)
        # d_x,d_y = extract_xy(denied_points,points_data)
        # i_x = [user_data[i][1] for i in 1:length(user_data)]
        # i_y = [user_data[i][2] for i in 1:length(user_data)]
        # plot_image([i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], "data/test.png")
    end
    return user_points, accepted_points, denied_points
end

user_points, accepted_points, denied_points = _run()

u_x,u_y = extract_xy(user_points,points_data)
a_x,a_y = extract_xy(accepted_points,points_data)
d_x,d_y = extract_xy(denied_points,points_data)
i_x = [user_data[i][1] for i in 1:length(user_data)]
i_y = [user_data[i][2] for i in 1:length(user_data)]

plot_image([i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename)

# ## Display Monte Carlo tree for first decision
#history = collect(stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3))
#a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
#inchrome(D3Tree(info[:tree], init_expand=3))

println("Done")