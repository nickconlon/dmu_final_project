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

function beta(s, a)
    if a == "wait"
        return [0,0,0]
    elseif a == "building"
        return [1,0,0]
    elseif a == "road"
        return [0,1,0]
    else
        return [0,0,1]
    end
end


function sample_initial_state(rng)
    phi = randn(rng, 3)
    return State(phi)
end

struct State
    phi::Vector{Float64}
    # history of points
end

m = QuickPOMDP(
    #states = ["building", "road", "other"],
    actions = ["building", "road", "other", "wait"],
    observations = ["building", "road", "other", "accept", "deny"],
    initialstate = ImplicitDistribution(sample_initial_state), # hidden state right here
    discount = 0.95,

    transition = function (s, a)
        if a == "wait"
            return Deterministic(s) # 
        else 
            return Deterministic(s) # 
        end
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
            p_accept = s.phi/sum(s.phi) # turn into probability. s.phi is continuous (x,y,z)
            p_deny = 1-p_accept
            # return SparseCat(["accept", "deny"], p)
        end
    end,

    reward = function (s, a)
        return dot(beta(s,a), s.phi)
    end
)

pomdp =  m
println(statetype(m))

pomdp = m
up = BootstrapFilter(pomdp, 1000)
solver = POMCPSolver(tree_queries=1000, c=100.0, rng=MersenneTwister(1), tree_in_info=true)
planner = solve(solver, pomdp)

# for (s, a, o, ai) in stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3)
#     println("State was $s,")
#     println("action $a was taken,")
#     println("and observation $o was received.\n")
#     println(typeof(ai))
# end


#history = collect(stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3))

a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
inchrome(D3Tree(info[:tree], init_expand=3))