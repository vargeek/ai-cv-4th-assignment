import util
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os


def getTranslateMatrix(dx, dy, dtype=np.float32):
    """
    获取位移矩阵
    """
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
    ], dtype=dtype)


def scaleImage(img, scale=0.3):
    """
    缩放图像
    """
    mat = cv2.getRotationMatrix2D((0, 0), 0, scale)
    rows, cols, *_ = img.shape
    return cv2.warpAffine(img, mat, (round(cols*scale), round(rows*scale)))


def warpPoint(M, pt):
    """
    计算 `pt` 坐标经过 `M` 投射后的坐标。M.shape: (3,3), pt.shape: (2,)
    """
    _x, _y, _w = M @ np.array([*pt, 1])
    _w = 1 / _w
    return _x * _w, _y * _w


def warpRect(M, rect):
    """
    计算 `rect` 经过 `M` 投射后的坐标。  
    `rect[0:2]` 为长方形左上角坐标，`rect[2:4]` 为右下角坐标
    """
    return [*warpPoint(M, rect[0:2]), *warpPoint(M, rect[2:4])]


def joinRects(rect1, rect2):
    """
    计算包含 `rect1` 和 `rect2` 的最小 `rect` 坐标
    """
    return [min(rect1[0], rect2[0]), min(rect1[1], rect2[1]), max(rect1[2], rect2[2]), max(rect1[3], rect2[3])]


def imageRect(img):
    """
    获取图像边框坐标
    """
    rows, cols, *_ = img.shape
    return [0, 0, cols, rows]


def imageMask(img):
    """
    获取图像内容遮罩。用于计算变换后的新遮罩，便于进行 `blend`
    """
    rows, cols, *_ = img.shape
    return np.ones((rows, cols), dtype=np.uint8) * 255


def blendImages(img1, mask1, img2, mask2):
    """
    混合两张图片  
    两张图片和遮罩尺寸都相等
    """
    rows, cols, *_ = img1.shape
    chs = []

    # 获取各个通道，分开进行混合，最后再合并
    chs_img1 = [*cv2.split(img1)]
    chs_img2 = [*cv2.split(img2)]
    for i in range(len(chs_img1)):
        ch1 = chs_img1[i]
        ch2 = chs_img2[i]
        ch = np.zeros((rows, cols), dtype=np.uint8)
        # 第一张图片
        index = mask1 > 0
        ch[index] = ch1[index]
        # 只包含第二张图片的位置
        index = (mask2 > 0) & (mask1 == 0)
        ch[index] = ch2[index]
        # 两张图片重叠的部分
        index = (mask2 > 0) & (mask1 > 0)
        ch[index] = ch[index] * 0.5 + ch2[index] * 0.5
        chs.append(ch)
    return cv2.merge((*chs,))


def stitching(img1, img2):
    """
    拼接两张图片
    """
    mask1 = imageMask(img1)
    mask2 = imageMask(img2)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)

    # 先检测两张图片的特征点
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 暴力匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return None

    # 画出50对匹配的 `keypoints`
    # matches_img = cv2.drawMatches(
    #     img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('matches_img', matches_img)

    img1_pts = np.array([kp1[m.queryIdx].pt for m in matches])
    img2_pts = np.array([kp2[m.trainIdx].pt for m in matches])

    # 用 `RANSAC` 计算变换矩阵
    M, _ = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC, 5.0)

    # 计算新画布的大小
    rect1 = imageRect(img1)
    rect2 = imageRect(img2)
    rect2 = warpRect(M, rect2)
    rect = joinRects(rect1, rect2)
    cols = int(round(rect[2] - rect[0]))
    rows = int(round(rect[3] - rect[1]))

    # 第一张图在新画布中的位移
    translate = getTranslateMatrix(-rect[0], -rect[1])

    # 把两张图和遮罩投射到新画布中
    warped_img1 = cv2.warpAffine(img1, translate, (cols, rows))
    warped_mask1 = cv2.warpAffine(mask1, translate, (cols, rows))

    warped_img2 = cv2.warpPerspective(img2, M, (cols, rows))
    warped_img2 = cv2.warpAffine(warped_img2, translate, (cols, rows))
    warped_mask2 = cv2.warpPerspective(mask2, M, (cols, rows))
    warped_mask2 = cv2.warpAffine(warped_mask2, translate, (cols, rows))
    # cv2.imshow('warped_img1', warped_img1)
    # cv2.imshow('warped_img2', warped_img2)
    # cv2.imshow('warped_mask1', warped_mask1)
    # cv2.imshow('warped_mask2', warped_mask2)

    # 使用变换后的图片和遮罩混合两张图
    result = blendImages(warped_img1, warped_mask1, warped_img2, warped_mask2)
    # cv2.imshow('result', result)
    return result


def stitchingWithFilepaths(file1, file2, scale=0.3):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    if img1 is None or img2 is None:
        return None

    img1 = scaleImage(img1, scale)
    img2 = scaleImage(img2, scale)
    return stitching(img1, img2)


def _test():
    pairs = [
        ['img01.jpg', 'img03.jpg'],
        ['img01.jpg', 'img05.jpg'],
        ['img01.jpg', 'img07.jpg'],
        ['img01.jpg', 'img09.jpg'],
    ]
    for names in pairs:
        file1 = util.findFile(names[0])
        file2 = util.findFile(names[1])
        result = stitchingWithFilepaths(file1, file2)
        if result is not None:
            cv2.imshow('{}-{}'.format(*names), result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # _test()

    if len(sys.argv) != 3:
        print('请输入两个图片地址，第一张图为左边的图')
        exit(1)

    scale = 0.3
    result = stitchingWithFilepaths(sys.argv[1], sys.argv[2], scale)
    if result is not None:
        cv2.imshow('result', result)
        print("选中窗口后点击任意键退出")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("输入图片有误")
