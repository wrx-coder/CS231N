import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    #config = 某个参数专属的优化器参数+历史状态
    #setdefault:
    #如果 config 里已经有 "learning_rate"，就保持原值不动
    #没有就加上1e-2
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    ##############################################################
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
      ###############################################################
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config["momentum"] * v - config["learning_rate"] * dw
    #如果很多步梯度方向差不多，v 会越来越稳定，优化速度更快。如果某一步梯度有噪声，历史速度会帮你“抹平”这种抖动。
    #减小抖动，加速收敛
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
#普通 SGD / Momentum 有一个共同问题：所有参数都用同一个学习率 lr，但不同参数的梯度尺度可能差很多
#结果会出现：某些维度更新太猛（震荡），某些维度更新太慢（学不动）
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)#表示“历史梯度平方平均”保留多少过去信息。
                                         #越大（如 0.99）：更平滑，更稳定，越小（如 0.9）：更敏感，变化更快
    config.setdefault("epsilon", 1e-8)   #用于防止除零
    config.setdefault("cache", np.zeros_like(w))#这是 RMSProp 的核心状态变量：和 w 同形状，存“梯度平方的滑动平均”

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    cache = config["cache"]
    decay = config["decay_rate"]
    eps = config["epsilon"]
    lr = config["learning_rate"]

    cache = decay * cache + (1 - decay) * (dw ** 2)
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    config["cache"] = cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    #Adam 同时维护两个“历史量”：
    #m：梯度的滑动平均（像 momentum）
    #v：梯度平方的滑动平均（像 RMSProp）
    #偏差修正（因为一开始 m 和 v 都从 0 开始，前几步会偏小）
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)  #梯度均值
    config.setdefault("beta2", 0.999)#梯度平方均值
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)        #第几次更新（标量计数器）

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    config["t"] += 1
    t = config["t"]
    
    lr = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    eps = config["epsilon"]
    
    m = config["m"]
    v = config["v"]
    
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * dw
    
    # Update biased second moment estimate
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    
    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Parameter update
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # Save back to config
    config["m"] = m
    config["v"] = v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
