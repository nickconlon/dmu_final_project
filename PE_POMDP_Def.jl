using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, QMDP
using POMDPModels
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
using StaticArrays

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

function sample_initial_state(m)
    # Determine the starting state. Calls the global state variable and puts it in a Vector
    # Can be modified to add noise
    POI = m.user_points  
    avg_b = mean([POI[a][4] for a in 1:length(POI)])
    avg_r = mean([POI[a][5] for a in 1:length(POI)])
    avg_n = mean([POI[a][6] for a in 1:length(POI)])
    cov_b = std([POI[a][4] for a in 1:length(POI)])
    cov_r = std([POI[a][5] for a in 1:length(POI)])
    cov_n = std([POI[a][6] for a in 1:length(POI)])
    phi = [avg_b,avg_r,avg_n] 
    phi = phi/norm(phi)  # Normalize
    init_cov = [cov_b^2 0 0; 0 cov_r^2 0; 0 0 cov_n^2]
    return State(phi,init_cov,[],0)
end

mutable struct PE_POMDP <: POMDP{SVector{3,Float64},Array{Float64,2},Int64}
    user_points::Array{Any,1}
    user_accuracy::Float64
    user_availability::Float64
    discount_factor::Float64
end

#--- POMDP Definition ---#
struct State
    phi::Vector{Float64}
    cov::Array{Float64,2}    
    acts::Any
    step::Int64
end

function POMDPs.initialstate(m::PE_POMDP)
    POI = m.user_points  
    avg_b = mean([POI[a][4] for a in 1:length(POI)])
    avg_r = mean([POI[a][5] for a in 1:length(POI)])
    avg_n = mean([POI[a][6] for a in 1:length(POI)])
    cov_b = std([POI[a][4] for a in 1:length(POI)])
    cov_r = std([POI[a][5] for a in 1:length(POI)])
    cov_n = std([POI[a][6] for a in 1:length(POI)])
    phi = [avg_b,avg_r,avg_n] 
    phi = phi/norm(phi)  # Normalize
    init_cov = [cov_b^2 0 0; 0 cov_r^2 0; 0 0 cov_n^2]
    init_state = State(phi,init_cov,[],0)
    return ImplicitDistribution(init_state)
end

POMDPs.discount(pomdp::PE_POMDP) = pomdp.discount_factor

function POMDPs.transition(::PE_POMDP,s,a)
    s.step += 1  # Increment step counter
    s.acts = push!(s.acts,a)
    return Deterministic(s) # The state never changes
end

function POMDPs.reward(pomdp::PE_POMDP,s,a)
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
end

# Observation Definition

POMDPs.observations(::PE_POMDP) = make_observations()  #How to tie in history?

function POMDPs.observation(m::PE_POMDP,s::State,a,sp)
    if a == "wait"
        # points already accepted should get zero probability mass
        # [p1, p2,... pn] = (p(p1), p(p2)) = normalized(sim_metric(p1, s.phi), sim_metric(p2, s.phi), ...)
        p_a = []
        acts = []
        for (i, act) in enumerate(actions(m,b))  # How does this work?
            if act != "wait"
                b = beta_values[parse(Int64,act)]#(x,y,z)
                out = create_state()
                sim_metric = similarity(out[1], b)
                if sim_metric < 0.5
                    sim_metric = 0.0001
                else
                    sim_metric = sim_metric*sim_metric
                end
                push!(p_a,sim_metric)
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
        sim_metric = similarity(s.phi, beta_values[parse(Int64,a)])
        sim_metric = sim_metric*0.8
        acc = m.user_accuracy
        av = m.user_availability
        # sim_metric = sim_metric/norm(sim_metric)

        p = [av*(acc*sim + (1-acc)*(1-sim)),av*(acc*(1-sim)*(1-acc)*sim)]
        return SparseCat(["accept", "deny"], p)
    end

end

# POMDPs.actions(::PE_POMDP) = make_actions()

function POMDPs.actions(m::PE_POMDP,b)
    # Creates a list of potential actions in an array of strings
    # states = b.hist
    println(b)
    total_act = length(points_data)
    acts = Array{String}(undef, total_act+1)
    prev_a = [h[1] for h in s.acts]  #s.acts is a tuple with (action,observation)
    for a in 1:total_act
        if ~any(i -> string(i)==i,prev_a) 
            acts[a] = string(a)
        end
    end
    acts[end] = "wait"
    return acts
end

function POMDPs.isterminal(::PE_POMDP,s) 
    if s.step == T+1
        return true
    else
        return false
    end
end

PE_fun =  PE_POMDP(user_road,0.9,0.8,0.99)  # Define POMDP
up = BootstrapFilter(PE_fun, 1000)  # Unweighted particle filter
solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
planner = solve(solver, PE_fun)
history = collect(stepthrough(PE_fun, planner, up, "s,a,o,action_info", max_steps=3))
a, info = action_info(planner, initialstate(PE_fun), tree_in_info=true)
inchrome(D3Tree(info[:tree], init_expand=3))