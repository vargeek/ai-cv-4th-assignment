import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def show_image_gray(filepath):
    """
    显示灰度图片，打印图片基本信息
    """
    img_gray = cv2.imread(filepath, 0)
    cv2.imshow('lenna_gray', img_gray)

    # 图片数据
    print(img_gray)

    # 每个像素8bit
    print(img_gray.dtype)

    # 512*512
    print(img_gray.shape)


def read_image(filepath):
    """
    读取彩色图片
    """
    img = cv2.imread(filepath)
    print(img)
    print(img.shape)
    return img


def crop_image(img):
    """
    切图
    """
    img_crop = img[0:100, 0:200]
    cv2.imshow('crop_image', img_crop)


def color_split(img):
    """
    分别显示RGB灰度图
    """
    B, G, R = cv2.split(img)
    cv2.imshow('lenna_B', B)
    cv2.imshow('lenna_G', G)
    cv2.imshow('lenna_R', R)


def random_light_color_channel(channel):
    """
    随机亮化一个通道
    """
    rand = random.randint(-50, 50)
    if rand == 0:
        pass
    elif rand > 0:
        lim = 255 - rand
        channel[channel > lim] = 255
        channel[channel <= lim] = (
            rand + channel[channel <= lim]).astype(channel.dtype)
    else:
        lim = 0 - rand
        channel[channel < lim] = 0
        channel[channel >= lim] = (
            rand + channel[channel >= lim]).astype(channel.dtype)


def random_light_color(img):
    """
    随机亮化图片
    """
    B, G, R = cv2.split(img)
    random_light_color_channel(B)
    random_light_color_channel(G)
    random_light_color_channel(R)
    img_merge = cv2.merge((B, G, R))
    return img_merge


def change_color(img):
    """
    亮化并显示图片
    """
    img_random_color = random_light_color(img)
    cv2.imshow('img_random_color', img_random_color)


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma

    table = [(x / 255.0)**invGamma * 255 for x in range(256)]
    table = np.array(table).astype("uint8")
    return cv2.LUT(img, table)


def gamma_correction(img_dark):
    """
    gamma校正
    """
    img_brighter = adjust_gamma(img_dark, 2)
    cv2.imshow('img_dark', img_dark)
    cv2.imshow('img_brighter', img_brighter)
    return img_brighter


def show_histogram(img):
    """
    equalize the histogram of the Y channel
    """
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    # uncomment next line to show the hist window, then click the close button to continue.
    # plt.show()

    shape = img.shape
    smaller = cv2.resize(img, (int(shape[0]*0.5), int(shape[1]*0.5)))

    img_yuv = cv2.cvtColor(smaller, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    # y: luminance(明亮度), u&v: 色度饱和度
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imshow('Color input image', smaller)
    cv2.imshow('Histogram equalized', img_output)


def show_rotation(img):
    """
    show rotated image
    """
    M = cv2.getRotationMatrix2D(
        (img.shape[1]/2, img.shape[0]/2), 30, 1)  # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna', img_rotate)
    print("rotate matrix: \n", M)

    img_rotate2 = cv2.warpAffine(img_rotate, M, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna2', img_rotate2)

    M3 = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 30, 0.5)
    print("rotate matrix 3: \n", M3)

    img_rotate3 = cv2.warpAffine(img_rotate, M3, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna3', img_rotate3)


def show_affine_transform(img):
    """
    affine transform 
    """
    rows, cols, _ = img.shape
    src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows - 1]])
    dst_pts = np.float32(
        [[cols*0.2, rows*0.1], [cols*0.9, rows*0.2], [cols*0.1, rows*0.9]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('affine lenna', dst)


def random_corner_points(w, h, delta):
    pts = [
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ]

    def rand_val(x):
        if x == 0:
            return random.randint(-delta, delta)
        else:
            return random.randint(x-delta, x)
    pts = [[rand_val(pt[0]), rand_val(pt[1])] for pt in pts]

    return pts


def random_perspective_transform_matrix(shape):
    h, w, _ = shape

    src_pts = np.float32(random_corner_points(w, h, 60))
    dst_pts = np.float32(random_corner_points(w, h, 60))

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def show_perspective_transform(img):
    """
    perspective transform
    """
    shape = img.shape
    h, w, _ = shape
    matrix = random_perspective_transform_matrix(shape)

    img_warp = cv2.warpPerspective(img, matrix, (w, h))

    cv2.imshow('perspective lenna', img_warp)


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(proj_dir), 'assets')

    filepath = os.path.join(assets_dir, 'lenna.jpg')

    # 显示灰度图片
    show_image_gray(filepath)
    cv2.waitKey(0)  # 按任意键退出

    # 读取彩色图片
    img = read_image(filepath)
    cv2.imshow('lenna', img)
    cv2.waitKey(0)

    # 剪切图片
    crop_image(img)
    cv2.waitKey(0)

    # 分别显示RGB灰度图
    color_split(img)
    cv2.waitKey(0)

    # 亮化并显示图片
    change_color(img)
    cv2.waitKey(0)

    # gamma校正
    img_brighter = gamma_correction(img)
    cv2.waitKey(0)

    # histogram
    show_histogram(img_brighter)
    cv2.waitKey(0)

    # rotate
    show_rotation(img)
    cv2.waitKey(0)

    # affine transform
    show_affine_transform(img)
    cv2.waitKey(0)

    # perspective transform
    show_perspective_transform(img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
