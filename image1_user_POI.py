import cv2
import matplotlib.pyplot as plt

input_array = cv2.imread("images/Image1_raw.png")

# Case 1: Front doors (Building/road)
user_frontdoor = [[544,1316],[619,1447],[827,1668],[1193,1793],[1482,1926]]
# Case 2: Edges of roads
user_roadedges = [[1864,1775],[779,1120],[1075,1660],[570,1135],[1032,1165]]
# Case 3: Intersection of roads
user_roadintersection = [[934,1733],[1843,1251],[1138,1784],[1324,2025],[1554,878]]
# Case 4: Building corners
user_corners = [[494,1244],[375,1314],[673,1829],[1152,1573],[1565,1824]]
# Case 5: Back doors (Building/other)
user_backdoor = [[1430,1510],[1502,1693],[750,1836],[420,1490],[583,1811]]
# Case 6: Just road
user_road = [[734,1357],[883,1625],[1367,930],[1403,329],[1832,1526]]
# Case 7: Just building
user_building = [[454,1314],[578,1533],[1321,1610],[1414,1786],[1306,1725]]
# Case 8: Just other
user_other = [[524,756],[1100,733],[1778,435],[1466,1257],[598,2003]]

#Plotting
plt.imshow(input_array)
for p in user_other:
    plt.scatter(p[0], p[1], 10, c="red")
# plt.show()
plt.savefig("images/user_other")

