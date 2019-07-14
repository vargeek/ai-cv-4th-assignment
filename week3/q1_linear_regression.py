import random
import matplotlib.pyplot as plt
import numpy as np


def inference(Theta, X):
    """
    Theta: (n+1, 1), `n` 为 `variable` 个数  
    X: shape: (nb_samples, n+1)  
    retval: (nb_samples, 1)
    """
    return X @ Theta


def eval_loss(Theta, X, Ygt):
    """
    Theta: (n+1, 1)  
    X: (nb_samples, n+1)  
    Ygt: (nb_samples, 1)
    """
    nb_samples = len(X)
    diff = (X @ Theta) - Ygt
    avg_loss = (diff.transpose() @ diff)[0][0] / nb_samples
    return avg_loss


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


def train(X, Ygt, batch_size, lr, max_iter):
    """
    X: (nb_samples, n+1)  
    Ygt: (nb_samples, 1)  
    """
    Theta = np.array([[0, 0]], dtype=X.dtype).transpose()
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
    theta_0 = random.random() * 5
    theta_1 = random.random() * 10

    nb_samples = 100
    xs = [random.random() * 10 for _ in range(nb_samples)]
    ys = [theta_0 + theta_1 * x + random.random() * 2 - 1 for x in xs]

    return xs, ys, theta_0, theta_1


def gen_sample_matrix(nb_samples=100):
    xs, ys, theta_0, theta_1 = gen_sample_data()

    dtype = np.float
    X = np.array([xs], dtype=dtype).transpose()
    One = np.ones((nb_samples, 1))
    X = np.hstack((One, X))

    Y = np.array([ys], dtype=dtype).transpose()

    Theta = np.array([[theta_0, theta_1]], dtype=dtype).transpose()
    return X, Y, Theta


def draw(X, Y, Theta, loss):
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle("q1_linear_regression")

    plt1.set_xlabel("x")
    plt1.set_ylabel("y")
    plt1.scatter(X[:, 1], Y[:, 0], label="samples")
    x = np.linspace(0, 10, 100)
    y = Theta[0][0] + x * Theta[1][0]
    plt1.plot(x, y, label="h_theta", color='red')
    plt1.legend()

    plt2.plot(list(range(len(loss))), loss, label='loss')
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.show()


def run():
    nb_samples = 100
    learn_rate = 0.001
    max_iter = 100

    X, Y, Theta = gen_sample_matrix(nb_samples)
    print("X: {}, Y: {}, Theta: {}".format(X.shape, Y.shape, Theta.shape))

    model, loss = train(X, Y, 50, learn_rate, max_iter)

    print('Theta:{} \n{}\ntrain:{} \n{}'.format(
        Theta.shape, Theta, model.shape, model))

    draw(X, Y, model, loss)


if __name__ == "__main__":
    run()
