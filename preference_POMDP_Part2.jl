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

function sample_initial_state(rng)
    # Determine the starting state
    # Modified to be rand instead of randn
    # Can be modified to take in the distribution of initial points
    POI = user_data
    #Take mean of observed points and add noise
    avg_b = mean([POI[a][4] for a in 1:length(POI)])+rand(rng)/10
    avg_r = mean([POI[a][5] for a in 1:length(POI)])+rand(rng)/10
    avg_n = mean([POI[a][6] for a in 1:length(POI)])+rand(rng)/10
    phi = [avg_b,avg_r,avg_n]
    phi = phi/norm(phi) # Normalize
    return State(phi, [])
end

struct State
    phi::Vector{Float64}
    history::Vector{String}
end

function make_observations()
    total_act = length(points_data)
    a = Array{String}(undef, total_act+2)
    for i in 1:total_act
        a[i] = string(i)
    end
    a[end-1] = "accept"
    a[end] = "deny"
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

m = QuickPOMDP(
    #states = ["building", "road", "other"],
    actions = make_actions(),
    observations = make_observations(),
    initialstate = ImplicitDistribution(sample_initial_state), # hidden state distribution
    discount = 0.95,

    transition = function (s, a)
        return Deterministic(s) # The state never changes
    end,

    observation = function (s, a, sp)
        if a == "wait"
            best = argmax(s.phi)
            p = [0.1,0.1,0.1]
            p[best] = 0.8

            # points already accepted should get zero probability mass
            # [p1, p2,... pn] = (p(p1), p(p2)) = normalized(sim(p1, s.phi), sim(p2, s.phi), ...)
            
            # get point p greatest sim
            # s.history.push!(p)
            # return SparseCat([p], [1])
            # else

            A = []
            for b in enumerate(beta_values)
                idx = b[1]
                beta = b[2] #(x,y,z)
                sim = similarity(s.phi, beta)
                push!(A,sim)
            end
            A = A/norm(A)
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
for (s, a, o, ai) in stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end


# ## Display Monte Carlo tree for first decision
# a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
# inchrome(D3Tree(info[:tree], init_expand=3))
history = collect(stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3))
a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
inchrome(D3Tree(info[:tree], init_expand=3))

