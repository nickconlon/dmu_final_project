import matplotlib.pyplot as plt
import cv2

raw_image = cv2.imread(
    "/home/hunter/Documents/Research/Preference_Elicitation/dmu_final_project/images/neighborhood_image.jpg")
plt.imshow(raw_image)


def plot_points(points, color):
    for p in points:
        plt.scatter(p[0], p[1], 15, c=color)


# Building
building_pts = [[114, 99], [294, 86], [451, 82], [447, 393], [606, 411]]
# Road
road_pts = [[44, 253], [215, 232], [620, 227], [493, 250], [653, 255]]
# Other
other_pts = [[153, 484], [246, 492], [140, 28], [388, 162], [502, 480]]
# Frontdoor
frontdoor_pts = [[138, 339], [294, 340], [467, 142], [645, 144], [625, 337]]
# Backdoor
backdoor_pts = [[70, 444], [73, 54], [428, 48], [635, 450], [425, 441]]
# Road Edges
road_edges_pts = [[215, 276], [385, 275], [551, 274], [51, 275], [396, 204]]
# Building Corners
corner_pts = [[46, 60], [73, 337], [215, 447], [547, 461], [557, 119], [508, 341]]
# Intersections
intersection_pts = [[268, 297], [121, 187], [446, 186], [652, 294], [160, 278]]

plot_points(building_pts,"red")
plot_points(road_pts,"blue")
plot_points(other_pts,"black")
plot_points(frontdoor_pts,"purple")
plot_points(backdoor_pts,"orange")
plot_points(road_edges_pts,"cyan")
plot_points(corner_pts,"pink")
plot_points(intersection_pts, "green")
plt.show()
