using GLMakie
using Distributions

test_d = Dirichlet([2,2.01,1])
s = rand(test_d,1000)
co = LinRange(0,3,1000)
meshscatter(s[1,:],s[2,:],s[3,:],markersize = 0.03,color = co)