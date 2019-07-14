import random
import matplotlib.pyplot as plt
import numpy as np


def inference(Theta, X):
    """
    Theta: (n+1, 1), `n` 为 `variable` 个数  
    X: shape: (nb_samples, n+1)  
    retval: (nb_samples, 1)
    """
    z = X @ Theta
    return 1 / (1 + np.exp(-z))


def gradient(Ypred, Ygt, X):
    """
    Ypred: (nb_samples, 1) 
    Ygt: (nb_samples, 1)
    X: (nb_samples, n+1)
    retval: (n+1, 1)
    """
    nb_samples = len(X)
    d = X.transpose() @ (Ypred - Ygt) / nb_samples
    return d


def cal_step_gradient(X, Ygt, Theta, lr):
    """
    X: (nb_samples, n+1)  
    Ygt: (nb_samples, 1)  
    Theta: (n+1, 1)  
    lr: learn rate
    retval: (n+1, 1)
    """
    Ypred = inference(Theta, X)
    d = gradient(Ypred, Ygt, X)
    Theta = Theta - lr * d
    return Theta


def eval_loss(Theta, X, Ygt):
    """
    Theta: (n+1, 1)
    X: (nb_samples, n+1)
    Ygt: (nb_samples, 1)
    """
    Ypred = inference(Theta, X)  # (m, 1)
    Ygt_T = Ygt.transpose()  # (1,m)
    loss = -Ygt_T @ np.log(Ypred) - (1 - Ygt_T) @ np.log(1 - Ypred)
    avg_loss = loss[0][0] / len(X)

    return avg_loss


def train(X, Ygt, batch_size, lr, max_iter):
    """
    X: (nb_samples, n+1)  
    Ygt: (nb_samples, 1)  
    """
    Theta = np.array([[0, 0, 0]], dtype=X.dtype).transpose()
    nb_samples = len(X)
    losss = []
    for _ in range(max_iter):
        batch_indices = np.random.choice(nb_samples, batch_size)
        X_batch = X[batch_indices, :]
        Y_batch = Ygt[batch_indices, :]
        Theta = cal_step_gradient(X_batch, Y_batch, Theta, lr)
        loss = eval_loss(Theta, X, Ygt)
        losss.append(loss)
    return Theta, losss


def gen_sample_data(nb_samples=100):
    b = 5 + random.random() * 2 - 1  # [-4,6)
    k = -1 + random.random() * 1 - 0.5  # [-1.5,1.5)

    xs1 = [random.random() * 10 for _ in range(nb_samples)]
    xs2 = [random.random() * 10 - 5 for _ in range(nb_samples)]

    def random_y(x1, x2):
        delta = 0.5
        boundary = b + k * x1 + random.random() * delta - delta * 0.5
        return x2 >= boundary and 1 or 0

    ys = [random_y(xs1[i], xs2[i]) for i in range(nb_samples)]

    return xs1, xs2, ys, k, b


def gen_sample_matrix(nb_samples=100):
    xs1, xs2, ys, k, b = gen_sample_data()

    dtype = np.float
    X = np.array([xs1, xs2], dtype=dtype).transpose()
    One = np.ones((nb_samples, 1))
    X = np.hstack((One, X))

    Y = np.array([ys], dtype=dtype).transpose()

    return X, Y, k, b


def draw(X, Y, model, loss):
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle("q2_logistic_regression")

    plt1.set_xlabel("x1")
    plt1.set_ylabel("x2")
    pos_samples = (Y == 1).transpose()[0]
    neg_samples = pos_samples == False
    pos = X[pos_samples, :]
    neg = X[neg_samples, :]

    plt1.scatter(pos[:, 1], pos[:, 2], label="positive", color="red")
    plt1.scatter(neg[:, 1], neg[:, 2], label="negative", color="yellow")

    theta0 = model[0][0]
    theta1 = model[1][0]
    theta2 = model[2][0]
    xs1 = np.linspace(0, 10, 100)
    xs2 = []

    if theta2 != 0:
        xs2 = [-(theta0+theta1*x)/theta2 for x in xs1]
    plt1.plot(xs1, xs2, label="h_theta", color='blue')

    plt2.plot(list(range(len(loss))), loss, label='loss')
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.show()


def run():
    nb_samples = 100
    learn_rate = 0.03
    max_iter = 1000

    X, Y, k, b = gen_sample_matrix(nb_samples)
    print("X: {}, Y: {}, k: {}, b: {}".format(X.shape, Y.shape, k, b))

    model, loss = train(X, Y, 50, learn_rate, max_iter)

    draw(X, Y, model, loss)


if __name__ == "__main__":
    run()
