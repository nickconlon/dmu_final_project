import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import autoencoder_feature_extractor as ae_extractor
import bgr_feature_extractor as class_extractor


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


def show_save_image(img, data, fname, show=False, save=True):
    """
    Save the image
    :param img:     The image to save
    :param data:    Any sampled data to overlay on the image of the form [px, py, radius, features...]
    :param fname:   The filename for the saved image
    :param show:    True if we should show the image
    :param save:    True of we should save the image
    """
    im = img.copy()
    for i, d in enumerate(data):
        im = cv.circle(im, (d[0], d[1]), d[2], color=(255, 255, 255), thickness=1)
        im = cv.putText(im, str(i), (d[0], d[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    if save:
        cv.imwrite(fname, im)

    if show:
        plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        plt.show()


def sample(im, extractor, sample_points=None, num_points=50, r=20):
    """
    Sample uniformly at random num_points samples with radius r and feature vector v=(f1, f2,...,fn)

    :param im:              The image to sample
    :param extractor:       The feature extractor to use
    :param sample_points:   A pre-sampled set of points to use
    :param num_points:      The number of points to sample
    :param r:               The radius of each point

    :return:                list[point_x, point_y, r, f1, f2,...,fn]
    """

    # radius must be an integer
    r = int(r)

    w = im.shape[1]
    h = im.shape[0]

    # The masking stuff is just to make the sample a circle with radius r
    mask = np.zeros([2*r, 2*r], dtype=np.uint8)
    mask = cv.circle(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)+1), r+1, color=(255, 255, 255), thickness=-1)
    mask[mask == 0] = 1

    points = []
    for i in range(num_points):
        if sample_points is not None:
            pt = np.array(sample_points[i])
        else:
            pt = np.array([np.random.randint(r, w-r), np.random.randint(r, h-r)])

        t = im[pt[1]-r:pt[1]+r, pt[0]-r:pt[0]+r]
        try:
            masked_img = cv.bitwise_or(t, t, mask=mask)
        except:
            print("ERROR: pt={}, r={}".format(pt, r))
            continue

        class_prob = extractor.extract(masked_img, r)
        points.append([pt[0], pt[1], r, *class_prob])

    return points


def sample_ae(im, num_samples, r):
    """
    Sample using the AutoEncoder

    :param im:          The image to sample
    :param num_samples: The number of samples
    :param r:           The radius of each sample
    :return:            list[point_x, point_y, r, f1, f2,...,fn]
    """
    extractor = ae_extractor.Extractor(option='load')
    return sample(im=im, extractor=extractor, num_points=num_samples, r=r)


def sample_class(im, num_samples, r):
    """
    Sample using the %class based method

    :param im:          The image to sample
    :param num_samples: The number of samples
    :param r:           The radius of each sample
    :return:            list[point_x, point_y, r, f1, f2,...,fn]
    """
    extractor = class_extractor.Extractor()
    return sample(im=im, extractor=extractor, num_points=num_samples, r=r)

###
#
# Below here is sampling functions for writing CSVs..
#
###
def sample_known_points(extractor, idx, radius_px=50, show_img=False, save_img=True, write_file=False):
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
    IMAGE_DIR = '../images/'
    DATA_DIR = '../data/'

    for d in data:
        SAMPLES = list(d.values())[0]
        img_name = 'airstrip_hand_segmented.png'
        CSV_NAME = list(d.keys())[0]

        # load image
        img = cv.imread(IMAGE_DIR+img_name)

        # TODO sample uniformly over classes, not (x,y) points
        samples = np.array(SAMPLES)
        samples = samples.astype(int)
        data = sample(img, extractor=extractor, sample_points=samples, num_points=len(samples), r=radius_px)
        if write_file:
            csv_fname = DATA_DIR+CSV_NAME+'_'+str(len(SAMPLES))+'_'+img_name.split('.')[0]+'_'+extractor.type()+'.csv'
            write(csv_fname, data, ["x", "y", "r", *extractor.feature_labels()])

        if save_img:
            image_fname = DATA_DIR+CSV_NAME+'_'+str(len(SAMPLES))+'_'+img_name.split('.')[0]+'_'+extractor.type()+'.png'
            show_save_image(img, data, image_fname, show=show_img, save=save_img)


def sample_random_image_data(extractor, idx, img_name, samples=300, radius_px=50, show_img=False, save_img=True, write_file=False):
    """
    Randomly sample points on the image located at ../images/<img_name> and optionally write to CSV file, save, or plot

    :param extractor:   The extractor to sample with
    :param idx:         Some index we can append to the csv (in case this is in a loop)
    :param img_name:    The filename of the image
    :param samples:     The number of samples to take
    :param radius_px:   The radius of each sample
    :param show_img:    True if the image should be shown w/ samples annotated
    :param save_img:    True if the image shoule be saved w/ samples annotated
    :param write_file:  True if the CSV should be written out
    """
    IMAGE_DIR = '../images/'
    DATA_DIR = '../data/'

    # load image
    img = cv.imread(IMAGE_DIR+img_name)

    data = sample(img, extractor=extractor, sample_points=None, num_points=samples, r=radius_px)
    if write_file:
        csv_fname = DATA_DIR+"sampled_"+str(samples)+'_'+img_name.split('.')[0]+'_'+extractor.type()+'_'+str(idx)+'.csv'
        write(csv_fname, data, ["x", "y", "r", *extractor.feature_labels()])

    if save_img:
        image_fname = DATA_DIR+"sampled_"+str(samples)+'_'+img_name.split('.')[0]+'_'+extractor.type()+ '.png'
        show_save_image(img, data, image_fname, show=show_img, save=save_img)


if __name__ == "__main__":
    #sample_from_data()
    # Standard BGR class feature extractor
    #extractor = class_extractor.Extractor()

    # Cool NN feature extractor
    #extractor = nn_extractor.Extractor()

    # Even cooler AE extractor
    extractor = ae_extractor.Extractor(option='load')
    sample_radius_px = 50
    images = ['airstrip_hand_segmented.png', 'neighborhood_image_segmented.png']

    sample_random_image_data(extractor, 1, images[0], radius_px=75, samples=600, show_img=True, write_file=True)
    sample_random_image_data(extractor, 1, images[1], radius_px=25, show_img=True)

    sample_random_image_data(extractor, 2, images[0], radius_px=25, samples=600, show_img=True, write_file=True)

    sample_known_points(extractor, 1, radius_px=sample_radius_px)
