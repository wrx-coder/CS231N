from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    #reg:正则化强度
    #C(class)类
    #D(dimension)特征维度
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0             #从0开始累加损失
    dW = np.zeros_like(W)  #初始化梯度矩阵

    # compute the loss and the gradient
    num_classes = W.shape[1]  #C类别数
    num_train = X.shape[0]    #N样本数
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores) #最大的分数变成 0，其它分数 ≤ 0，防止指数爆炸
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        
        loss -= logp[y[i]]  # negative log probability is the loss

        
        #solution part1
        dscores = p.copy()
        dscores[y[i]] -= 1
            #损失函数对原始得分的导数可得结论
            # accumulate gradient for W
        for j in range(num_classes):
            dW[:, j] += X[i] * dscores[j]
        
        #solution part1

    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # 边算梯度，边更新Loss                                                        
    #############################################################################
    
    #solution part2
    dW /= num_train
    dW += 2 * reg * W
    #soulution part2
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)#(N D) (D,C)==(N,C)
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)#概率矩阵(N C)
    N=X.shape[0]
    correct_class_probs = probs[np.arange(N),y]                        
    #假设y={1,0,2},correct_class_probs等价于probs[[0, 1, 2], [1, 0, 2]]
    #等价于取出probs矩阵里的[0,1],[1,0],[2,2]
    #即“选对”的概率
    loss = -np.sum(np.log(correct_class_probs))/N + 0.5*reg*np.sum(W**2) 
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dscores = probs.copy()
    dscores[np.arange(N),y] -=1
    dscores /=N
    dw = X.T.dot(dscores) + reg*W
    
    return loss, dW
    #在 CS231n 里反复看到这两个版本，本质是在训练你两种能力
    #循环版（naive）：你会不会把数学公式翻成代码
    #向量化版（vectorized）：你会不会把代码再翻成矩阵运算（高效）
    