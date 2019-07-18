import util
import cv2
import numpy as np
import random
import math


if __name__ == "__main__":
    # 读取原始图片和旋转后的图片
    img1 = cv2.imread(util.getLennaFilepath())
    img2 = cv2.imread(util.getLenna2Filepath())

    # 使用ORB算法检测 `keypoints`
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # # 显示 `keypoints`
    # kp1img = cv2.drawKeypoints(img1, kp1, None, color=(
    #     0, 255, 0), flags=0)
    # kp2img = cv2.drawKeypoints(img2, kp2, None, color=(
    #     0, 255, 0), flags=0)
    # cv2.imshow('img1kp', kp1img)
    # cv2.imshow('img2kp', kp2img)

    # match 两张图片的 `keypoints`
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 画出20对匹配的 `keypoints`
    matches_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    result_img = matches_img

    src_pts = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # matchesMask = mask.ravel().tolist()

    cv2.imshow('result_img', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
