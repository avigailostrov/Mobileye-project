import numpy as np

from model_1 import *
from SFM_2 import *
from SFM_standAlone import *
from second import *
#from dataset import *
#from neural_net import *


"""def init():
    build_dataset()
    build_net()"""


def first(img):
    path = r'C:\Users\user\Desktop\bootcamp\mobileye\mobileyeProject'
    img_r, img_g = find_tfl_lights(path + "\\" + img)
    red_x, red_y = max_filter(img_r, 1.3, 60)
    colors = []
    candidates = list(zip(red_x, red_y))
    for i in range(len(candidates)):
        colors += "red"
    green_x, green_y = max_filter(img_g, 1.3, 60)
    candidates += list(zip(green_x, green_y))
    size = len(candidates) - len(colors)
    for i in range(size):
        colors += "green"
    image = mpimg.imread(path + "\\" + img)
    plt.imshow(image)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)
    plt.show()
    return candidates, colors


def second(img, candidates, colors):
    traffic_lights = []
    new_colors = []
    path = r"C:\Users\user\Desktop\bootcamp\mobileye\mobileyeProject\candidates"
    files = glob.glob(path)
    #for f in files:
        #os.remove(f)
    counter = 0
    for candidate in candidates:
        x = candidate[0]
        y = candidate[1]
        my_path = crop_image(x, y, img, counter)
        if is_tfl(my_path):
            traffic_lights += [candidate]
            new_colors += colors[counter]
        counter += 1
    return traffic_lights, new_colors


def third(prev_img, curr_img, prev_frame_id, curr_frame_id, prev_tfl, curr_tfl, colors_1, colors_2): #EM, pp, focal
    prev_container = FrameContainer(prev_img)
    curr_container = FrameContainer(curr_img)
    with open('dusseldorf_000049.pkl', 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    p = [tuple([int(i), int(j)]) for i, j in prev_tfl]
    c = [tuple([int(i), int(j)]) for i, j in curr_tfl]
    prev_container.traffic_light = p
    curr_container.traffic_light = c
    EM = np.eye(4)
    for i in range(prev_frame_id, curr_frame_id):
        EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
    curr_container.EM = EM
    curr_container = SFM_2.calc_TFL_dist(prev_container, curr_container, focal, pp)
    visualize(prev_container, curr_container, focal, pp, prev_frame_id, curr_frame_id)
    """distances = np.array([42, 39, 46])
    return distances"""


def main():
    for i in range(24, 29):
        prev_candidates = []
        prev_colors = []
        curr_frame_id = i+1
        prev_frame_id = i
        pkl_path = 'dusseldorf_000049.pkl'
        prev_img_path = 'dusseldorf_000049_0000' + str(prev_frame_id) + '_leftImg8bit.png'
        curr_img_path = 'dusseldorf_000049_0000' + str(curr_frame_id) + '_leftImg8bit.png'
        if prev_frame_id == 24:
            prev_candidates, prev_colors = first(prev_img_path)
            #print(prev_candidates)
            prev_tfl, prev_colors = second(prev_img_path, prev_candidates, prev_colors)
        else:
            prev_tfl = curr_tfl
            prev_colors = curr_colors
        curr_candidates, curr_colors = first(curr_img_path)
        curr_tfl, curr_colors = second(curr_img_path, curr_candidates, curr_colors)
        prev_tfl = [(935.0, 168.0), (1118.0, 334.0), (547.0, 338.0), (869.0, 412.0)]
        curr_tfl = [(935.0, 168.0), (1118.0, 334.0), (542.0, 333.0), (869.0, 412.0)]
        third(prev_img_path, curr_img_path, prev_frame_id, curr_frame_id, prev_tfl, curr_tfl, prev_colors, curr_colors,
              )


if __name__ == '__main__':
    #init()
    main()