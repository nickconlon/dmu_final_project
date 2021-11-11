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
        d = [int(x) for x in d]
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
        #plt.imshow(t)
        #plt.show()
        try:
            masked_img = cv.bitwise_or(t, t, mask=mask)
        except:
            print("ERROR: pt={}, r={}".format(pt, r))
            #continue
            masked_img = t

        class_prob = extractor.extract(masked_img, r)
        points.append([pt[0], pt[1], r, *class_prob])

    return np.array(points)


def sample_ae(im, num_samples, r, sample_points=None):
    """
    Sample using the AutoEncoder

    :param im:              The image to sample
    :param num_samples:     The number of samples
    :param r:               The radius of each sample
    :param sample_points    List of points to sample or None if random  sample
    :return:                list[point_x, point_y, r, f1, f2,...,fn]
    """
    extractor = ae_extractor.Extractor(option='load')
    return sample(im=im, extractor=extractor, sample_points=sample_points, num_points=num_samples, r=r)


def sample_class(im, num_samples, r, sample_points=None):
    """
    Sample using the %class based method

    :param im:              The image to sample
    :param num_samples:     The number of samples
    :param r:               The radius of each sample
    :param sample_points    List of points to sample or None if random  sample
    :return:                list[point_x, point_y, r, f1, f2,...,fn]
    """
    extractor = class_extractor.Extractor()
    return sample(im=im, extractor=extractor, sample_points=sample_points, num_points=num_samples, r=r)


class dummy_user_point_msg:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class dummy_image_msg:
    def __init__(self, img):
        self.img = img

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


def sample_random_image_data_combined(img_name, samples=300, radius_px=50, show_img=False, save_img=True, write_file=False):
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
    DATA_DIR = '../data/combined_features/'

    # load image
    img = cv.imread(IMAGE_DIR+img_name)
    w = img.shape[1]
    h = img.shape[0]
    xs = np.random.randint(radius_px, w-radius_px, size=samples)
    ys = np.random.randint(radius_px, h-radius_px, size=samples)

    extractor_ae = ae_extractor.Extractor(option='load')
    extractor_class = class_extractor.Extractor()
    class_feature_len = len(extractor_class.feature_labels())
    ae_feature_len = len(extractor_ae.feature_labels())

    sample_points = [[x,y] for x,y in zip(xs, ys)]

    data_ae = sample(img, extractor=extractor_ae, sample_points=sample_points, num_points=samples, r=radius_px)
    data_class = sample(img, extractor=extractor_class, sample_points=sample_points, num_points=samples, r=radius_px)

    sampled_data = np.zeros([samples, 3+class_feature_len+ae_feature_len])
    sampled_data[:, 0:3] = data_ae[:, 0:3] # copy over (x,y,r)
    sampled_data[:, 3:3+class_feature_len] = data_class[:, 3:] # copy over class features
    sampled_data[:, 3+class_feature_len:3+class_feature_len+ae_feature_len] = data_ae[:, 3:] # copy over ae features

    sampled_data[sampled_data == 0.0] = 1e-5

    if write_file:
        csv_fname = DATA_DIR+"sampled_"+str(samples)+'_'+img_name.split('.')[0]+'_combined'+'.csv'
        write(csv_fname, sampled_data, ["x", "y", "r", *extractor_class.feature_labels(), *extractor_ae.feature_labels()])

    if save_img:
        image_fname = DATA_DIR+"sampled_"+str(samples)+'_'+img_name.split('.')[0]+'_'+extractor.type()+ '.png'
        show_save_image(img, sampled_data, image_fname, show=show_img, save=save_img)


def sample_known_points_combined(d, radius_px=50, show_img=False, save_img=True, write_file=False):

    data_user = [
        {'user_frontdoor': [[544, 1316], [619, 1447], [761, 1701], [1256, 1760], [1482, 1926]]},
        {'user_roadedges': [[1864, 1775], [779, 1120], [1075, 1660], [570, 1135], [1456, 657]]},
        {'user_roadintersection': [[934, 1733], [1858, 1211], [1138, 1784], [1264, 2004], [1451, 787]]},
        {'user_corners': [[494, 1244], [375, 1314], [696, 1874], [1152, 1573], [1565, 1824]]},
        {'user_backdoor': [[1430, 1510], [1502, 1693], [750, 1836], [420, 1490], [583, 1811]]},
        {'user_road': [[734, 1357], [883, 1625], [1538, 809], [1009, 347], [1778, 1194]]},
        {'user_building': [[487, 1381], [578, 1533], [1321, 1610], [1414, 1786], [1306, 1725]]},
        {'user_other': [[524, 756], [1100, 733], [1731, 435], [1466, 1257], [598, 2003]]}
    ]
    # frontdoor   roadedges   roadint   corner   backdoor   road  building  other
    data_final = [
        {'final_frontdoor': [[138, 339], [294, 340], [467, 142], [645, 144], [625, 337]]},
        {'final_roadedges':  [[215, 276], [385, 275], [551, 274], [51, 275], [396, 204]]},
        {'final_roadintersection': [[268, 297], [121, 187], [446, 186], [652, 294], [160, 278]]},
        {'final_corners': [[46, 60], [73, 337], [215, 447], [547, 461], [557, 119], [508, 341]]},
        {'final_backdoor': [[70, 444], [73, 54], [428, 48], [635, 450], [425, 441]]},
        {'final_road': [[44, 253], [215, 232], [620, 227], [493, 250], [653, 255]]},
        {'final_building': [[114, 99], [294, 86], [451, 82], [447, 393], [606, 411]]},
        {'final_other': [[153, 484], [246, 492], [140, 28], [388, 162], [502, 480]]}
    ]
    all_data_final = [
        {'all_final': [[138, 339], [294, 340], [467, 142], [645, 144], [625, 337],
        [215, 276], [385, 275], [551, 274], [51, 275], [396, 204],
        [268, 297], [121, 187], [446, 186], [652, 294], [160, 278],
        [46, 60], [73, 337], [215, 447], [547, 461], [557, 119], [508, 341],
        [70, 444], [73, 54], [428, 48], [635, 450], [425, 441],
        [44, 253], [215, 232], [620, 227], [493, 250], [653, 255],
        [114, 99], [294, 86], [451, 82], [447, 393], [606, 411],
        [153, 484], [246, 492], [140, 28], [388, 162], [502, 480]]}
    ]
    if d == "final":
        data = all_data_final
        IMAGE_DIR = '../images/'
        DATA_DIR = '../data/combined_features/final_features/'
        img_name = 'neighborhood_image_segmented_r.png'
    else: # user features
        data = data_user
        IMAGE_DIR = '../images/'
        DATA_DIR = '../data/combined_features/user_features/'
        img_name = 'airstrip_hand_segmented.png'

    extractor_ae = ae_extractor.Extractor(option='load')
    extractor_class = class_extractor.Extractor()
    class_feature_len = len(extractor_class.feature_labels())
    ae_feature_len = len(extractor_ae.feature_labels())

    for d in data:
        SAMPLES = list(d.values())[0]
        CSV_NAME = list(d.keys())[0]

        # load image
        img = cv.imread(IMAGE_DIR+img_name)

        # TODO sample uniformly over classes, not (x,y) points
        samples = np.array(SAMPLES)
        #samples = np.flip(samples, axis=1)
        samples = samples.astype(int)

        data_ae = sample(img, extractor=extractor_ae, sample_points=samples, num_points=len(samples), r=radius_px)
        data_class = sample(img, extractor=extractor_class, sample_points=samples, num_points=len(samples), r=radius_px)

        sampled_data = np.zeros([len(samples), 3 + class_feature_len + ae_feature_len])
        sampled_data[:, 0:3] = data_ae[:, 0:3]  # copy over (x,y,r)
        sampled_data[:, 3:3 + class_feature_len] = data_class[:, 3:]  # copy over class features
        sampled_data[:, 3 + class_feature_len:3 + class_feature_len + ae_feature_len] = data_ae[:, 3:]  # copy over ae features

        sampled_data[sampled_data == 0.0] = 1e-5

        if write_file:
            csv_fname = DATA_DIR+CSV_NAME+'_'+str(len(SAMPLES))+'_'+img_name.split('.')[0]+'_combined'+'.csv'
            write(csv_fname, sampled_data, ["x", "y", "r", *extractor_class.feature_labels(), *extractor_ae.feature_labels()])

            summary_data_mean = np.mean(sampled_data[:,3:], axis=0)
            summary_data_mean[summary_data_mean == 0.0] = 1e-5
            summary_data_std = np.std(sampled_data[:,3:], axis=0)
            summary_data_std[summary_data_std == 0.0] = 1e-5
            summary_data = []
            for i in range(len(summary_data_mean)):
                summary_data.append([summary_data_mean[i], summary_data_std[i]])

            with open(csv_fname.replace('.csv', '_summary.csv'), "w") as file:
                file.write(",".join([*extractor_class.feature_labels(), *extractor_ae.feature_labels()]) + "\n")
                file.write(str(summary_data))

        if save_img:
            image_fname = DATA_DIR+CSV_NAME+'_'+str(len(SAMPLES))+'_'+img_name.split('.')[0]+'_combined'+'.png'
            show_save_image(img, sampled_data, image_fname, show=show_img, save=save_img)


if __name__ == "__main__":
    #sample_from_data()
    # Standard BGR class feature extractor
    #extractor = class_extractor.Extractor()

    # Cool NN feature extractor
    #extractor = nn_extractor.Extractor()

    # Even cooler AE extractor
    extractor = ae_extractor.Extractor(option='load')
    sample_radius_px = 50
    images = ['airstrip_hand_segmented.png', 'neighborhood_image_segmented_r.png']

    #sample_random_image_data_combined(images[0], radius_px=75, samples=300, show_img=True, write_file=True)
    #sample_random_image_data_combined(images[1], radius_px=20, samples=300, show_img=True, write_file=True)
    #sample_known_points_combined(d="final", radius_px=20, write_file=True, show_img=True)
    sample_known_points_combined(d="user", radius_px=75, write_file=True, show_img=True)


    #sample_random_image_data(extractor, 1, images[1], radius_px=25, show_img=True)
    #sample_random_image_data(extractor, 2, images[0], radius_px=25, samples=600, show_img=True, write_file=True)

