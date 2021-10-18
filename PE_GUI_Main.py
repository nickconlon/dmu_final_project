from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
import sys
import numpy as np
# import rospy


class PreferenceAlgorithm:
    def __init__(self):
        # Constant definition
        self.num_particles = 500    # Number of particles generated in the particle filter. Final value should ~= 4000
        self.p_sample = 10          # Number of points that the algorithm iterates over
        self.num_guess = 5          # Number of guesses performed by the algorithm before final propagation
        self.init_guess = 5         # Number of points before the algorithm starts running
        self.user_points = []       # Points selected by the user
        self.POMDP = []             # Current POMDP instantiation
        self.belief = []            # Belief structure
        self.phi = []               # Current best user estimate
        self.best_points_idx = []
        self.best_points_phi = []
        self.guess_points = []      # Points that can be suggested by the algorithm
        self.guess_beta = []        # Points that can be suggested by the algorithm. Beta values only
        self.final_points = []      # Points that can be propagted by the algorithm
        self.final_beta = []        # Points that can be propagated. Beta values only
        self.observe_points = []    # Random sample of points that could be seen by the algorithm
        self.observe_beta = []      # Random sample of points that could be seen by the algorithm. Beta values only
        self.suggest = []           # Last suggestion done by the algorithm
        self.suggested = []         # Points suggested by the algorithm
        # Initialize ROS message passing
        # self.point_sub = rospy.Subscriber("/gui_points",point,user_points)
        # self.user_obs = rospy.Subscriber("/user_response",Bool,PF_update)
        # self.seg_image = rospy.Subscriber("/seg_image",??,image_sample)
        # self.suggest_pub = rospy.Publisher("/suggest",point,queue_size=1)
        # self.populate_pub = rospy.Publisher("/populate",point)

        # Call and run Julia functions
        Main.include("PE_POMDP_Def.jl")
        Main.include("ParticleFilter_Def.jl")
        self.user = Main.user_expert
        self.solver = Main.POMCPSolver(tree_queries=100, c=100.0, rng=Main.MersenneTwister(1), tree_in_info=True,
                                       estimate_value=Main.FORollout(Main.RandomSolver()))
        print(self.solver)
        print("Algorithm Initialization Complete")

    def new_user_point(self, msg):
        """Function adds new user points to a list.
        Will keep adding items to a list until full number of guesses has been reached"""
        self.user_points.append(msg)
        if len(self.user_points) == self.init_guess:
            # Generate a mean vector to initialize particle belief
            self.phi = [sum(x) / len(x) for x in zip(*self.user_points)]
            # Initialize belief
            self.belief = Main.init_PF(self.phi, self.num_particles)

    def generate_suggestion(self):
        """Function is given a set of user points and generates a suggestion based on the algorithm
        Engage after user clicks box -- for now..."""
        self.phi = [sum(x) / len(x) for x in zip(*self.belief.states)]
        # Sample set of points to iterate over
        self.best_points_idx, self.best_points_phi = Main.find_similar_points(self.guess_beta, self.phi,
                                                                              self.p_sample, self.suggested)
        # Define POMDP with new set of points
        self.POMDP = Main.PE_POMDP(self.user_points, self.best_points_phi, self.observe_beta, self.final_beta,
                                   self.user, 0.99, self.num_guess+1)
        # Define solver
        planner = Main.solve(self.solver, self.POMDP)
        # Extract expected suggestion
        suggest, info = Main.action_info(planner, Main.initialstate(self.POMDP), tree_in_info=False)
        self.suggested.append(suggest)
        self.num_guess -= 1     # Decrement step counter
        # Find index that the algorithm suggested
        if suggest != "wait":
            ans = self.guess_points[int(self.suggested[-1])]
            # Publish to ROS
            self.suggest_pub.publish(ans)
        # If the algorithm chooses to wait, then don't publish anything

    def belief_update(self, msg):
        self.belief = Main.update_PF(self.belief, self.POMDP, self.suggest, msg)
        return self.belief

    def sample_image(self, msg):
        """Function will take in a numpy array for the segmented image and populate necessary lists for sampling"""
        # Do image processing

        # Update
        self.guess_points = []  # Points that can be suggested by the algorithm
        self.final_points = []  # Points that can be propagted by the algorithm
        self.observe_points = []  # Random sample of points that could be seen by the algorithm

        # Sample beta values for...
        self.guess_beta = []  # Points that can be suggested by the algorithm. Beta values only
        self.final_beta = []  # Points that can be propagated. Beta values only
        self.observe_beta = []  # Random sample of points that could be seen by the algorithm. Beta values only


    def propagate_belief(self):
        """Applies point onto final image"""
        pass

# Feed image into processing
# Point =  [x,y,z]
# Output set of points for consideration

# Receive user points. Call function to output point for suggestion

# Receive user response. Function call for PF update

# Call populate points function

## MAIN FUNCTION INITIALIZATION SEQUENCE FOR JULIA #####
if __name__ == '__main__':
    PreferenceAlgorithm()
    # while not rospy.is_shutdown():
    #     rospy.spin()
