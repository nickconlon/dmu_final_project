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

include("data_read.jl")
random_data = read_data("data/random_data.csv")
user_road = read_data("data/user_road.csv")

obs_to_idx = Dict("building"=>1, "road"=>2, "other"=>3)

function beta(s, a)
    #Return a 1-hot vector reflective of the observation. 
    #Is is similar towards asking "How much do you care about this class?"
    
    # zero rewards for waiting
    if a == "wait"
        return [0,0,0]
    end

    # negative rewards if the agent is suggesting something wrong
    if argmax(s.phi) != obs_to_idx[a]
       return [-2,-2,-2]
    end

    # positive rewards if the agent is suggesting something right
    if a == "building"
        return [1,0,0]
    elseif a == "road"
        return [0,1,0]
    else # other
        return [0,0,1]
    end
end

function sample_initial_state(rng)
    # Determine the starting state
    # Modified to be rand instead of randn
    # Can be modified to take in the distribution of initial points
    # phi = rand(rng, 3)
    POI = user_road
    #Take mean of observed points and add noise
    avg_b = mean([POI[a][4] for a in 1:length(POI)])+rand(rng)/10
    avg_r = mean([POI[a][5] for a in 1:length(POI)])+rand(rng)/10
    avg_n = mean([POI[a][6] for a in 1:length(POI)])+rand(rng)/10
    phi = [avg_b,avg_r,avg_n]
    return State(phi)
end

struct State
    phi::Vector{Float64}
    # history of points
end

function make_observations()
    a = Array{String}(undef, 102)
    for i in 1:100
        a[i] = string(i)
    end
    a[101] = "accept"
    a[102] = "deny"
    return a
end

function make_actions()
    a = Array{String}(undef, 102)
    for i in 1:100
        a[i] = string(i)
    end
    return a
    # deleteat!(a, findall(x->x=="1", a))
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
            return SparseCat(["building", "road", "other"], p)
        else
            # agent is suggesting 'a'
            # operator likes s.phi
            # suggest the max element (what does this mean?) of 'a' with probability 
            # agent is suggesting 'a', operator likes s.phi
            if argmax(s.phi) == obs_to_idx[a]
                return SparseCat(["accept", "deny"], [0.9, 0.1])
            else
                return SparseCat(["accept", "deny"], [0.1, 0.9])
            end


        end
    end,

    reward = function (s, a)
        r = dot(beta(s,a), s.phi)
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

