import util
import cv2
import numpy as np
import random
import math

epsilon = 1e-7


def ransacMatching(A, B):
    """
    A: List of List
    B: List of List
    """
    dtype = A.dtype
    niters = 1000
    maxGoodCount = 0
    goodH = None

    iter = 0
    while iter < niters:
        iter = iter + 1
        points = randomPoints(A, B)
        if points == None:
            continue
        (a, b, *_) = points
        a = np.array(a, dtype=dtype)
        b = np.array(b, dtype=dtype)
        H = computeH(a, b)
        goodCount = findInliersCount(A, B, H)

        if goodCount > maxGoodCount:
            maxGoodCount = goodCount
            goodH = H
            # niters = min(1000, updateIterNb(goodCount/len(A)))

    return goodH


def ransacMatching2(A, B):
    """
    A: List of List
    B: List of List
    """
    niters = 1000
    maxGoodCount = 0
    goodH = None

    inlierA = A[:4]
    inlierB = B[:4]

    for _ in range(0, niters):
        H = computeH(inlierA, inlierB)
        (goodCount, inlierA, inlierB) = findInliers(A, B, H)

        if goodCount > maxGoodCount:
            maxGoodCount = goodCount
            goodH = H
            # niters = min(1000, updateIterNb(goodCount/len(A)))
        elif goodCount == maxGoodCount:
            break
    return goodH


def findInliersCount(A, B, H, threshold=3):
    """
    寻找内点个数
    """
    err = computeReprojError(A, B, H)
    goodCount = list(err < threshold).count(True)
    return goodCount


def findInliers(A, B, H, threshold=3):
    """
    寻找内点
    """
    err = computeReprojError(A, B, H)
    mask = err < threshold
    a = A[mask]
    b = B[mask]
    return (len(a), a, b)


def computeH(M, m):
    """
    计算H变换矩阵
    """
    count = len(M)
    dtype = M.dtype
    cM = np.sum(M, 0) / count
    cMs = np.array([cM], dtype=dtype).repeat(count, 0)
    cm = np.sum(m, 0) / count
    cms = np.array([cm], dtype=dtype).repeat(count, 0)

    sM = np.sum(abs(M - cMs), 0)
    sm = np.sum(abs(m-cms), 0)
    if (sM < epsilon).any() or (sm < epsilon).any():
        return None
    sm = count / sm
    sM = count / sM

    invHorm = np.array([
        [1/sm[0], 0, cm[0]],
        [0, 1/sm[1], cm[1]],
        [0, 0, 1],
    ], dtype=dtype)
    Hnorm2 = np.array([
        [sM[0], 0, -cM[0]*sM[0]],
        [0, sM[1], -cM[1]*sM[1]],
        [0, 0, 1],
    ], dtype=dtype)

    LtL = np.zeros((9, 9), dtype=dtype)

    for i in range(0, count):
        (x, y) = (m[i] - cm) * sm
        (X, Y) = (M[i] - cM) * sM
        Lx = np.array([[
            X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x
        ]], dtype=dtype)
        Ly = np.array([[
            0, 0, 0, X, Y, 1, -y*X, -y*Y, -y
        ]], dtype=dtype)

        LtL = LtL + Lx.transpose()@Lx + Ly.transpose()@Ly

    LtL = cv2.completeSymm(LtL)
    ok, _, V = cv2.eigen(LtL)
    if not ok:
        return None

    H0 = V[8].reshape((3, 3))
    HTemp = V[7].reshape((3, 3))
    HTemp = invHorm @ H0
    H0 = HTemp @ Hnorm2 * 1/H0[2, 2]
    return H0


def computeReprojError(A, B, H):
    """
    计算每个样本的误差
    """
    count = len(A)
    dtype = A.dtype
    err = np.zeros((count), dtype=dtype)
    for i in range(0, count):
        a = np.array([*A[i], 1], dtype=dtype)
        b = B[i]
        ww = 1 / H[2].dot(a)
        dx = H[0].dot(a) * ww - b[0]
        dy = H[1].dot(a) * ww - b[1]
        err[i] = dx * dx + dy * dy

    return err


def randomPoints(A, B, maxAttempts=300):
    """
    随机取出四对点
    """
    count = min(len(A), len(B))
    nbPts = 4
    indices = list(range(0, count))
    if count < nbPts:
        return None

    iter = 0
    while iter < maxAttempts:
        iter = iter + 1
        for idx in range(0, nbPts):
            _idx = random.randint(idx, count-1)
            indices[idx], indices[_idx] = indices[_idx], indices[idx]
        subIndices = indices[0:nbPts]
        a = [A[idx] for idx in subIndices]
        if not checkPoints(a):
            break

        b = [B[idx] for idx in subIndices]
        if not checkPoints(b):
            break
        return (a, b, subIndices)


def checkPoints(points):
    """
    检查每个点不在任意两个点的直线上
    """
    count = len(points)
    i = 0
    while i < count:
        j = 0
        while j < i:
            dx1 = points[j][0] - points[i][0]
            dy1 = points[j][1] - points[i][1]
            k = 0
            while k < j:
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]

                if abs(dx1*dy2-dx2*dy1) <= epsilon * (abs(dx1)+abs(dy1)+abs(dx2)+abs(dy2)):
                    break
                k = k + 1
            if k < j:
                break
            j = j + 1
        if j < i:
            break
        i = i + 1
    return i >= count


def updateIterNb(p_inlier, confidence=0.95, count=4):
    """
    计算推荐的迭代次数
    """
    confidence = max(epsilon, min(confidence, 1-epsilon))
    p_inlier = max(epsilon, min(p_inlier, 1-epsilon))
    a = math.log(1-confidence)
    b = math.log(1 - math.pow(p_inlier, count))
    return round(a/b)


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

    # 执行 `RANSAC` 算法，计算出变换矩阵 `H`
    A = np.array([kp1[m.queryIdx].pt for m in matches])
    B = np.array([kp2[m.trainIdx].pt for m in matches])
    H = ransacMatching(A, B)
    # H = ransacMatching2(A, B)

    # 画出原始图片的边框(红色)，以及该边框经过 `H` 变换得到的边框(蓝色)
    rows,cols,_ = img1.shape
    pts = np.array([[0,0],[0,rows-1],[cols-1,rows-1],[cols-1,0] ], dtype=np.float32).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    dst[:,:,0] += cols

    result_img = cv2.polylines(matches_img,[np.int32(pts)],True,(0,0,255),3, cv2.LINE_AA)
    result_img = cv2.polylines(result_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    cv2.imshow('result_img', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
