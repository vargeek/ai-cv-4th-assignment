import os
import sys
import cv2
import random
import numpy as np


def _randint(delta):
    """
    在`[-delta, delta]`内随机取一个整数
    """
    return random.randint(-delta, delta)


def _rand(delta):
    """
    在`[0, delta)`内随机取一个浮点数
    """
    return random.random() * delta


def _randfloat(a, b):
    """
    在`[a, b)`内随机取一个浮点数
    """
    return random.random() * (b - a) + a


def _rand_degrees(delta):
    """
    三个通道分别在`[-delta, delta]`范围内取一个随机值
    """
    b = _randint(delta)
    g = _randint(delta)
    r = _randint(delta)
    return (b, g, r)


def clamp(val, smallest, largest):
    """
    把`val`限制在`[smallest, largest]`范围内
    """
    return max(smallest, min(val, largest))


def crop_image(img, crop_rect=[0, 0, 1, 1]):
    """
    裁剪图片  
    param crop_rect: 左上角和右下角坐标。[x1, y1, x2, y2]，范围为[0,1]。
    """
    shape = img.shape
    rows, cols, _ = shape
    col1, row1, col2, row2 = crop_rect
    col1 = clamp(round(cols * col1), 0, cols)
    row1 = clamp(round(rows * row1), 0, rows)
    col2 = clamp(round(cols * col2), 0, cols)
    row2 = clamp(round(rows * row2), 0, rows)
    return img[row1:row2, col1:col2]


def _light_channel_color(channel, degree):
    """
    亮化一个通道
    """
    if degree == 0:
        pass
    elif degree > 0:
        lim = 255 - degree
        channel[channel > lim] = 255
        channel[channel <= lim] = (
            degree + channel[channel <= lim]).astype(channel.dtype)
    else:
        lim = 0 - degree
        channel[channel < lim] = 0
        channel[channel >= lim] = (
            degree + channel[channel >= lim]).astype(channel.dtype)


def light_color(img, degrees=(50, 50, 50)):
    """
    亮化图片  
    param degrees: 三个通道亮化灰度随机值取值范围  
        如：通道B随机范围为：[-degrees[0],degrees[0]]
    """
    B, G, R = cv2.split(img)
    _light_channel_color(B, degrees[0])
    _light_channel_color(G, degrees[1])
    _light_channel_color(R, degrees[2])
    img_merge = cv2.merge((B, G, R))
    return img_merge


def gamma_correction(img, gamma=1.0):
    """
    gamma校正
    """
    invGamma = 1.0 / gamma
    table = [(x / 255.0)**invGamma * 255 for x in range(256)]
    table = np.array(table).astype("uint8")
    return cv2.LUT(img, table)


def equalize_y_hist(img):
    """
    equalize the histogram of the Y channel
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def rotate_image(img, center=(0.5, 0.5), angle=0, scale=1):
    """
    旋转图片  
    param center: 旋转锚点。  
    """
    rows, cols, _ = img.shape
    ox = cols * center[0]
    oy = rows * center[1]
    matrix = cv2.getRotationMatrix2D((ox, oy), angle, scale)
    rotated = cv2.warpAffine(img, matrix, (cols, rows))
    return rotated


def perspective_transform_image(img, dst_pts):
    """
    图片投射  
    param dst_pts: 从左上角开始，逆时针方向的四个目标坐标，范围为[0, 1]
    """
    rows, cols, _ = img.shape

    src_pts = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ]

    src_pts = np.float32([[p[0]*cols, p[1]*rows] for p in src_pts])
    dst_pts = np.float32([[p[0]*cols, p[1]*rows] for p in dst_pts])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed = cv2.warpPerspective(img, matrix, (cols, rows))

    return transformed


def random_perspective_transform_image(img, delta=0.08):
    """
    随机图片投射  
    param delta: 随机值为[0~delta]
    """

    dst_pts = [
        [_rand(delta), _rand(delta)],
        [_rand(delta), 1-_rand(delta)],
        [1-_rand(delta), 1-_rand(delta)],
        [1-_rand(delta), _rand(delta)],
    ]
    return perspective_transform_image(img, dst_pts)


def random_augmentation(img, crop=0.05, light=50, gamma=(1, 2), equalize=True, angle=(-5, 5), perspective=0.05):
    """
    data augmentation  
    param crop: 四条边向内裁剪比例的随机值范围: [0, crop)  
    param light: 三个通道灰度变化的随机值范围: [-light, light]  
    param gamma: gamma校正随机值范围: [gamma[0], gamma[1])  
    param equalize: 是否执行`equalize_y_hist`  
    param angle: 旋转角度随机范围: [angle[0], angle[1])  
    param perspective: 投射变换四个顶点向内移动比例的随机范围: [0,perspective)
    """
    if type(img) == str:
        img = cv2.imread(img)

    # 裁剪
    if crop != None:
        crop_rect = [
            _rand(crop),    # TopLeftX: [0,crop)
            _rand(crop),    # TopLeftY: [0,crop)
            1-_rand(crop),  # BottomRightX: 1-[0,crop)
            1-_rand(crop),  # BottomRightY: 1-[0,crop)
        ]
        img = crop_image(img, crop_rect=crop_rect)

    # 亮化
    if light != None:
        degrees = (_randint(light), _randint(light), _randint(light))
        img = light_color(img, degrees)

    # gamma校正
    if gamma != None:
        img = gamma_correction(img, _randfloat(*gamma))

    # equalizeHist y
    if equalize:
        img = equalize_y_hist(img)

    # rotate
    if angle != None:
        img = rotate_image(img, angle=_randfloat(*angle), scale=0.9)

    # perspective_transform
    if perspective != None:
        img = random_perspective_transform_image(img, perspective)

    return img


def demo(filepath):
    img = cv2.imread(filepath)
    print('image shape: ', img.shape)
    cv2.imshow('lenna', img)

    # cropped = crop_image(img, [0, 0, 0.5, 1])
    # print('cropped shape: ', cropped.shape)
    # cv2.imshow('crop1', cropped)

    # cropped = crop_image(img, [0, 0, 1, 0.5])
    # print('cropped shape: ', cropped.shape)
    # cv2.imshow('crop2', cropped)
    # cv2.waitKey(0)

    # lighter = light_color(img, _rand_degrees(50))
    # cv2.imshow('light', lighter)
    # cv2.waitKey(0)

    # correction = gamma_correction(img, 2.0)
    # cv2.imshow('gamma_correction', correction)
    # cv2.waitKey(0)

    # equalized = equalize_y_hist(img)
    # cv2.imshow('equalize y hist', equalized)
    # cv2.waitKey(0)

    # rotated = rotate_image(img, angle=10, scale=0.5)
    # cv2.imshow('rotate', rotated)
    # cv2.waitKey(0)

    # transformed = random_perspective_transform_image(img)
    # cv2.imshow('transformed', transformed)
    # cv2.waitKey(0)

    augmentated = random_augmentation(img)
    print('augmentated shape: ', augmentated.shape)
    cv2.imshow('augmentated', augmentated)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(proj_dir), 'assets')
    filepath = os.path.join(assets_dir, 'lenna.jpg')
    # demo(filepath)

    files = sys.argv[1:]
    if len(files) == 0:
        files = [filepath for _ in range(0, 5)]

    files = [f for f in files if os.path.isfile(f)]
    for (i, f) in enumerate(files):
        img = random_augmentation(f)
        cv2.imshow("img_{}".format(i), img)

    if len(files) > 0:
        cv2.waitKey(0)
    cv2.destroyAllWindows()
