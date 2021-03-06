# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
# import os


# %%

def initial_centroids_kmeanspp(df, k):
    """
    K-Means++获取初始中心
    df: [nb_samples,['x,'y']]
    """
    nb_samples = len(df)

    # 随机选择一个中心
    idx = np.random.randint(0, nb_samples)
    # 被选为中心的样本索引值
    indices = [idx]

    # > D(x): the shortest distance from a data point to the closest center we have already chosen.
    D = None

    for _ in range(k - 1):
        # 使用上轮选中的中心更新每个样本到已选中心的最短距离
        c = df.loc[idx]
        newD = square_of_distance(df, c)
        if D is None:
            D = newD
        else:
            # 更新最短距离
            D[newD < D] = newD[newD < D]

        # 按概率随机选中一个样本
        p_acc = (D / np.sum(D)).cumsum()

        p = np.random.rand()
        # 选择`>=p`的最左边的值，可自动排除已选样本
        idx = min(binary_search(p_acc, p), nb_samples - 1)
        indices.append(idx)

    return df.loc[indices, ['x', 'y']].reset_index()


def binary_search(nums, val):
    """
    二分查找(`>=val`的最左边的值)
    """
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] < val:
            l = m + 1
        else:
            r = m - 1
    return l


def square_of_distance(pts1, pts2):
    """
    点距离的平方
    """
    # (x1 - x2)^2 - (y1 - y2)^2
    return (pts1.x - pts2.x)**2 + (pts1.y - pts2.y)**2


def cal_distance(pts1, pts2):
    """
    计算点距离
    """
    return np.sqrt(square_of_distance(pts1, pts2))


def assignment(df, centroids):
    """
    计算每个样本所属中心，存储到 `df.closest` 列
    """
    distance = pd.DataFrame({
        i:
        cal_distance(df, centroids.loc[i]) for i in centroids.index
    })
    df['closest'] = distance.idxmin(axis=1)
    return df


def update(df, centroids):
    """
    更新 `k` 个中心的坐标
    """
    for i in centroids.index:
        centroids.loc[i] = np.mean(df[df.closest == i])

    return centroids


def eval_loss(df, centroids):
    """
    计算损失
    """
    loss = 0
    for i in centroids.index:
        samples = df[df.closest == i]
        loss = loss + np.sum(square_of_distance(samples, centroids.loc[i]))

    return loss / len(df)


def draw(df, centroids, color_map, title):
    """
    绘制样本以及中心
    """
    fig = plt.figure(title)
    plot = fig.subplots()

    color = df.closest.map(lambda x: color_map[x])
    plot.scatter(df.x, df.y, color=color, alpha=0.5, edgecolors='k')

    plot.scatter(centroids.x, centroids.y, color=color_map, linewidths=6)
    plot.set_xlim(0, 80)
    plot.set_ylim(0, 80)

    # fig.savefig(os.path.join(os.path.dirname(
    #     os.path.realpath(__file__)), 'result', title) + '.png')


def draw_loss(loss, title, xlabel='step'):
    """
    绘制损失曲线
    """
    fig = plt.figure(title)
    plot = fig.subplots()
    xs = np.array(range(len(loss))) + 1

    plot.plot(xs, loss)
    plot.set_xlabel(xlabel)
    plot.set_ylabel('loss')

    # fig.savefig(os.path.join(os.path.dirname(
    #     os.path.realpath(__file__)), 'result', title) + '.png')


def get_color_map(k):
    """
    获取`k`个颜色值
    """
    colors = []
    for name in mcd.XKCD_COLORS:
        colors.append(mcd.XKCD_COLORS[name])
        if len(colors) >= k:
            break

    if len(colors) < k:
        low = 0.2
        high = 1.0
        for _ in range(k - len(colors)):
            colors.append(np.random.rand(3) * (high - low) + low)
    return colors


def run_kmeans(df, k, use_kmeanspp=False, should_draw=True, max_step=10):
    """
    执行`k-means`算法  
    df: 样本  
    k: `k`类  
    use_kmeanspp: 是否使用`k-means++`初始化中心点  
    should_draw: 是否绘制图片  
    max_step: 最大迭代次数  
    """
    color_map = get_color_map(k)

    # 初始化中心点
    centroids = None
    if use_kmeanspp:
        centroids = initial_centroids_kmeanspp(df, k)
    else:
        centroids = pd.DataFrame({
            'x': np.random.randint(0, 80, k),
            'y': np.random.randint(0, 80, k),
        })

    # 每个样本分配中心
    df = assignment(df, centroids)
    title = "k{}_{}".format(use_kmeanspp and 'pp' or '', k)
    if should_draw:
        draw(df, centroids, color_map, "{}_0".format(title))

    loss = [eval_loss(df, centroids)]

    for step in range(max_step):
        closest_centroids = df['closest'].copy(deep=True)

        # 更新中心坐标
        centroids = update(df, centroids)
        # 重新分配中心
        df = assignment(df, centroids)
        if should_draw:
            draw(df, centroids, color_map, "{}_{}".format(title, step+1))

        loss.append(eval_loss(df, centroids))
        if closest_centroids.equals(df['closest']):
            break

    if should_draw:
        draw_loss(loss, title+"_loss")
    return loss


# %%
if __name__ == "__main__":
    np.random.seed(0)

    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    # 运行kmeans
    run_kmeans(df, 3)
    # 运行kmeans++
    run_kmeans(df, 3, True)

    # 选择不同k值
    loss_of_k = []
    for k in range(10):
        loss = run_kmeans(df, k+1, True, False)
        loss_of_k.append(loss[-1])

    draw_loss(loss_of_k, 'loss_of_k', 'k')
    plt.show()
