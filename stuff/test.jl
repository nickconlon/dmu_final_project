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

numwins = 0

m = QuickPOMDP(
    #states = ["building", "road", "other"],
    actions = ["building", "road", "other", "wait"],
    observations = ["building", "road", "other"],
    initialstate = Uniform(["building", "road", "other"]),
    discount = 0.95,

    transition = function (s, a)
        if a == "wait"
            return Deterministic(s) # tiger stays behind the same door
        else # a door is opened
            return Uniform(["building", "road", "other"]) # reset
        end
    end,

    observation = function (s, a, sp)
        if a == "wait"
            # roll dice for not responding
            if sp == "building"
                return SparseCat(["building", "road", "other"], [0.9, 0.1, 0.1]) # sparse categorical distribution
            elseif sp == "road"
                return SparseCat(["road", "building", "other"], [0.9, 0.1, 0.1])
            else
                return SparseCat(["road", "building", "other"], [0.1, 0.1, 0.9])
            end
        else
            return SparseCat([s], [1]) #Uniform(["building", "road", "other"])
        end
    end,

    reward = function (s, a)
        if a == "wait"
            return -1.0
        elseif s == a # the tiger was found
            return 25.0
        else # the tiger was escaped
            return -50.0
        end
    end
)

pomdp =  m

# solver = QMDPSolver()
# policy = solve(solver, pomdp)

# plot(xlims=(0,1), 
# ylims=(minimum(Iterators.flatten(alphavectors(policy))), 
# maximum(Iterators.flatten(alphavectors(policy)))+50), 
# xlabel="belief", ylabel="alpha", title="SARSOP alpha vectors")

# println(policy)
# for v in alphavectors(policy)
#     println(v)
#     display(plot!([0,1, 2], v, label=v))
# end
#savefig("6-POMDPs/q1_sarsop_alpha.png")

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


history = collect(stepthrough(pomdp, planner, up, "s,a,o,action_info", max_steps=3))

a, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
inchrome(D3Tree(info[:tree], init_expand=3))