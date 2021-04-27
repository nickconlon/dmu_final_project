import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def write(file_name, d, labels):
    """
    Write out a csv. Each element in d is a line in the file.
    :param labels:      The csv labels in list of string
    :param file_name:   The filename to write out
    :param d:           The data to write out
    """
    with open(file_name, "w") as file:
        file.write(",".join(labels) + "\n")
        for dd in d:
            s_list = [str(elem) for elem in dd]
            s_list = ",".join(s_list) + "\n"
            file.write(s_list)


def sample(im, sample_points=None, num_points=50, r=20):
    """
    Sample uniformly at random num_points samples with radius r.
    class probability is (p1, p2, p3) = (building, road, nothing)

    :param im:              The image
    :param sample_points:   A sample of points to use
    :param num_points:      The number of points to sample
    :param r:               The radius of each point
    :return:                list[point_x, point_y, r, p1, p2, p3]
    """
    w = im.shape[1]
    h = im.shape[0]
    mask = np.zeros([2*r, 2*r], dtype=np.uint8)
    mask = cv.circle(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)+1), r+1, color=(255, 255, 255), thickness=-1)
    mask[mask == 0] = 1

    points = []
    for i in range(num_points):
        if sample_points is not None:
            pt = np.array(sample_points[i])
        else:
            pt = np.array([np.random.randint(r, w-r), np.random.randint(r, h-r)])
        too_close = False
        for p in points:
            if np.linalg.norm(p[0:2]-pt) <= 2*r:
                too_close = True
        if too_close:
            continue

        t = im[pt[1]-r:pt[1]+r, pt[0]-r:pt[0]+r]
        try:
            masked_img = cv.bitwise_or(t, t, mask=mask)
        except:
            print("ERROR: pt={}, r={}".format(pt, r))
            continue
        classes = np.zeros(3)
        for x in range(2*r):
            for y in range(2*r):
                rr = np.linalg.norm(np.array([r, r], dtype=np.uint8)-np.array([x,y]))
                if rr <= r:
                    c = masked_img[y, x]  # opencv uses BGR convention!
                    if c[0] >= 200:  # Blue
                        classes[1] += 1
                    elif c[2] >= 200:  # Red
                        classes[0] += 1
                    elif c[0] == 0 and c[1] == 0 and c[2] == 0:
                        classes[2] += 1

        class_prob = np.around(classes/sum(classes), 2)
        points.append([pt[0], pt[1], r, class_prob[0], class_prob[1], class_prob[2]])

        # Print out the image.
        im = cv.circle(im, (pt[0], pt[1]), r, color=(0, 255, 0), thickness=1)
        font = cv.FONT_HERSHEY_SIMPLEX
        org = (pt[0]-2*r, pt[1])
        font_scale = 0.5
        color = (255,255,255)
        thickness = 1
        cv.putText(im, str(list(class_prob)), org, font, font_scale, color, thickness, cv.LINE_AA)
    return points


def sample_from_data():
    # TODO change these before running!
    data = [
        {'user_frontdoor': [[544, 1316], [619, 1447], [827, 1668], [1193, 1793], [1482, 1926]]},
        {'user_roadedges': [[1864, 1775], [779, 1120], [1075, 1660], [570, 1135], [1032, 1165]]},
        {'user_roadintersection': [[934, 1733], [1843, 1251], [1138, 1784], [1324, 2025], [1554, 878]]},
        {'user_corners': [[494, 1244], [375, 1314], [673, 1829], [1152, 1573], [1565, 1824]]},
        {'user_backdoor': [[1430, 1510], [1502, 1693], [750, 1836], [420, 1490], [583, 1811]]},
        {'user_road': [[734, 1357], [883, 1625], [1367, 930], [1403, 329], [1832, 1526]]},
        {'user_building': [[454, 1314], [578, 1533], [1321, 1610], [1414, 1786], [1306, 1725]]},
        {'user_other': [[524, 756], [1100, 733], [1778, 435], [1466, 1257], [598, 2003]]}
    ]
    DATA = 1
    SAMPLES = list(data[DATA].values())[0]
    IMAGE_NAME = 'images/Image1_segmented.png'
    CSV_NAME = 'data/' + list(data[DATA].keys())[0]
    SCALE_FACTOR = 100
    RADIUS = 200

    # load image
    img = cv.imread(IMAGE_NAME)
    # scale image
    width = int(img.shape[1] * SCALE_FACTOR / 100)
    height = int(img.shape[0] * SCALE_FACTOR / 100)
    dim = (width, height)
    img = cv.resize(img, dim)
    radius = int(RADIUS * (SCALE_FACTOR / 100))

    # TODO sample uniformly over classes, not (x,y) points
    samples = np.array(SAMPLES) * (SCALE_FACTOR / 100)
    samples = samples.astype(int)
    data = sample(img, sample_points=samples, num_points=len(samples), r=75)
    write(CSV_NAME + ".csv", data, ["x", "y", "r", "building", "road", "none"])

    cv.imshow("", img)
    cv.waitKey(delay=100)
    plt.plot(0, 0, label="[building, road, none]", color='green')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.legend()
    plt.savefig(CSV_NAME + '.png')
    # plt.show()

def sample_raw():
    DATA = 8
    SAMPLES = None
    IMAGE_NAME = 'images/Image1_segmented.png'
    CSV_NAME = 'data/random_data_300'
    SCALE_FACTOR = 100
    RADIUS = 200

    # load image
    img = cv.imread(IMAGE_NAME)
    # scale image
    width = int(img.shape[1] * SCALE_FACTOR / 100)
    height = int(img.shape[0] * SCALE_FACTOR / 100)
    dim = (width, height)
    img = cv.resize(img, dim)
    radius = int(RADIUS * (SCALE_FACTOR / 100))

    # TODO sample uniformly over classes, not (x,y) points
    data = sample(img, sample_points=None, num_points=300, r=75)
    write(CSV_NAME + ".csv", data, ["x", "y", "r", "building", "road", "none"])

    cv.imshow("", img)
    cv.waitKey(delay=100)
    plt.plot(0, 0, label="[building, road, none]", color='green')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.legend()
    plt.savefig(CSV_NAME + '.png')
    plt.show()

if __name__ == "__main__":
    sample_from_data()