include("data_read.jl")

#Load point data
random_data = read_data("./data/sampled_600_airstrip_hand_segmented_ae16_1.csv")
random_data_300 = read_data("./data/sampled_600_airstrip_hand_segmented_ae16_1.csv")
neighborhood_data = read_data("./data/sampled_300_neighborhood_image_segmented_ae16_1.csv")

#User Data
user_frontdoor = read_data("./data/user_frontdoor_5_airstrip_hand_segmented_ae16.csv")
user_backdoor = read_data("./data/user_backdoor_5_airstrip_hand_segmented_ae16.csv")
user_building = read_data("./data/user_building_5_airstrip_hand_segmented_ae16.csv")
user_road = read_data("./data/user_road_5_airstrip_hand_segmented_ae16.csv")
user_other = read_data("./data/user_other_5_airstrip_hand_segmented_ae16.csv")
#test_points = read_data("./data/user_test_5_airstrip_hand_segmented_ae16.csv")
user_road_edges = read_data("./data/user_roadedges_5_airstrip_hand_segmented_ae16.csv")
points_data = random_data_300 #* (100/30)
final_points_data = neighborhood_data #