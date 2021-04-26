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

include("data_read.jl")
random_data = read_data("data/random_data.csv")
user_frontdoor = read_data("data/user_frontdoor.csv")
user_building = read_data("data/user_building.csv")
user_road = read_data("data/user_road.csv")
test_points = read_data("data/user_test.csv")

# Available points
points_data = test_points
# Points operator has chosen
user_data = user_building 



#Create beta Values
beta_values = [points_data[i][4:6] for i in 1:length(points_data)]

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
    POI = user_data
    #Take mean of observed points and add noise
    avg_b = mean([POI[a][4] for a in 1:length(POI)])
    avg_r = mean([POI[a][5] for a in 1:length(POI)])
    avg_n = mean([POI[a][6] for a in 1:length(POI)])
    return [avg_b,avg_r,avg_n]
end

global_phi = create_state()
i_var = 0.25
global_cov = [i_var^2 0 0; 0 i_var^2 0; 0 0 i_var^2]

function sample_initial_state(rng)
    # Determine the starting state
    # Modified to be rand instead of randn
    # Can be modified to take in the distribution of initial points
    POI = user_data
    #Take mean of observed points and add noise
    avg_b = global_phi[1]+rand(rng)/10
    avg_r = global_phi[2]+rand(rng)/10
    avg_n = global_phi[3]+rand(rng)/10
    phi = [avg_b, avg_r, avg_n]
    phi = phi/norm(phi) # Normalize
   
    init_cov = global_cov
    return State(phi,init_cov)
end

struct State
    phi::Vector{Float64}
    cov::Array{Float64,2}
end

function KalmanUpdate(b,o)
    #Assume that all observations update all classes
    o_var = 0.1
    cov_o = [o_var^2 0 0; 0 o_var^2 0; 0 0 o_var^2]

    #Time Update
    mu_p = b.phi  # Increment the mean
    cov_p = b.cov # Increment the covariance
    K = cov_p*I*inv(cov_p+cov_o)  #Kalman Gain

    #Measurement Update
    mu_b = mu_p + K*(o-mu_p)
    cov_b = (I-K)*cov_p

    return State([mu_b[1],mu_b[2],mu_b[3]],cov_b)
end

function make_observations()
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
    return 1-cosine_dist(x, y)
end

actions_list = make_actions()
observations_list = make_observations()
# initial_state = sample_initial_state(MersenneTwister(1))

m = QuickPOMDP(
    actions = actions_list,
    observations = observations_list,
    initialstate = ImplicitDistribution(sample_initial_state), # hidden state distribution
    discount = 0.95,

    transition = function (s, a)
        return Deterministic(s) # The state never changes
    end,

    observation = function (s, a, sp)
        if a == "wait"
            # points already accepted should get zero probability mass
            # [p1, p2,... pn] = (p(p1), p(p2)) = normalized(sim(p1, s.phi), sim(p2, s.phi), ...)
            A = []
            for (i, a) in enumerate(actions(m))
                if a != "wait"
                    b = beta_values[parse(Int64,a)]#(x,y,z)
                    sim = similarity(s.phi, b)
                    push!(A,sim)
                end
            end
            A = (length(A)>0 && sum(A)>0 ) ? A/norm(A) : A  #Catch cases where A is empty
            available_actions = actions(m)[1:end-1]
            return SparseCat(available_actions, A)
            # distribution over all operator add points
        else
            # agent is suggesting 'a'
            # operator likes s.phi
            # points already accepted should get denied
            sim = similarity(s.phi, beta_values[parse(Int64,a)])
            p = [sim, 1-sim]
            return SparseCat(["accept", "deny"], p)
        end
    end,

    reward = function (s, a)
        r = 0.0
        if a == "wait"
            r = 0.5
        else
            # if a was already accepted r = -5
            vec = beta_values[parse(Int64,a)] 
            r = similarity(s.phi, vec)
        end
        return r
    end
    #Note: No terminal state
)

pomdp =  m  # Define POMDP
up = BootstrapFilter(pomdp, 1000)  # Unweighted particle filter
solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
planner = solve(solver, pomdp)

#Step through simulates process
#actions_test
#println(observations(pomdp))
#deleteat!(actions_list, findall(x->x=="3", actions_list))
#deleteat!(observations_list, findall(x->x=="3", observations_list))
#println(observations(pomdp))
#vec = beta_values[3]
#deleteat!(beta_values, findall(x->x==vec, beta_values))


function _run()
    suggestions_received = [0]

    #for suggestions in 1:4
    while suggestions_received[1] < 2
        # update the state.phi values on before re initializing the POMCP
        #global_phi[1] = 0
        #global_phi[3] = 1
        pomdp =  m  # Define POMDP
        up = BootstrapFilter(pomdp, 1000)  # Unweighted particle filter
        solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
        planner = solve(solver, pomdp)
        println("----STEP----")
        for (s, a, o, ai) in stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3)
            println(s.phi)
            println("State was $s,")
            println("action $a was taken,")
            println("and observation $o was received.\n")
        
            # remove o if o is a point
            # remove a if o is accept
            println(observations_list)
            println(actions_list)
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
                break
            end

            if o == "deny"
                #If the user denies a suggestion, kick out potential point from action and observation space 
                deleteat!(actions_list, findall(x->x==a, actions_list))
                deleteat!(observations_list, findall(x->x==a, observations_list))

                #Update coveriance to increase uncertainty
                # new_cov= global_cov*1.2
                # new_cov = global_cov/similarity(a,global_phi)
                inv_beta = ones(3) - beta_values[parse(Int64,a)]
                new_state = KalmanUpdate(s,inv_beta)
                global_phi[1] = new_state.phi[1]
                global_phi[2] = new_state.phi[2]
                global_phi[3] = new_state.phi[3]
                global_cov[1:3] = new_state.cov[1:3]
                global_cov[4:6] = new_state.cov[4:6]
                global_cov[7:9] = new_state.cov[7:9]
                # println(global_cov)
                break
            end
            #If the user selects a point 
            if o != "accept" || o != "deny" || o != "no response"
                # remove o from actions & observations
                deleteat!(actions_list, findall(x->x==o, actions_list))
                deleteat!(observations_list, findall(x->x==o, observations_list))
            end
            if length(observations_list) <= 3 || length(actions_list) <= 1 || suggestions_received[1] >=2
                println("out of suggestions")
                break
            end
        end

    end
end

_run()

# ## Display Monte Carlo tree for first decision
#history = collect(stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3))
#a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
#inchrome(D3Tree(info[:tree], init_expand=3))

println("Done")