"""This file defines layer types that are commonly used for recurrent neural networks.
"""
import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b# @ 是矩阵乘法
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):#单步前向传播
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)输入
    - prev_h: Hidden state from previous timestep, of shape (N, H)上一步隐藏
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)把当前输入 x 投影到        hidden space 的权重
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    把上一时刻隐藏状态 prev_h 变换后再传给当前时刻的权重。

    它负责处理：

    历史记忆对当前隐藏状态的影响

    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    输入当前时刻的数据 x
	•	输入上一时刻的隐藏状态 prev_h
	•	输出当前时刻的隐藏状态 next_h
    """
    next_h = None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN.                 #
    ##############################################################################
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)  #注意不是x * Wx (逐元素乘法）
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
     rnn_forward 要做的事就是：
	1.	先拿初始隐藏状态 h0
	2.	依次处理第 1 个时间步、第 2 个时间步、…、第 T 个时间步
	3.	把每一步的 hidden state 都存起来
	4.	最后返回整个序列上的所有 hidden states
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    """
    #rnn_forward 本质上就是沿着时间维 T 循环调用 rnn_step_forward
    h = None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    h = []
    prev_h = h0
    T = x.shape[1]
    for t in range(T):
        x[:, t, :]
        #第一维 : 表示取所有样本,第二维 t 表示第 t 个时间步,第三维 : 表示该时间步整个输入向量
        #取出来(N, D)
        next_h = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h.append(next_h)
        prev_h = next_h
        
    h = torch.stack(h, dim=1)
    #把列表拼起来
    #假设有T个，每个shape(N,H)
    #拼接后变成(N, T, H)
        
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h


def word_embedding_forward(x, W):#词嵌入：把单词的整数编号，查表变成对应的词向量
    #eg: <START> a dog runs
    #计算机不会直接处理字符串，所以先会把它编码成整数：[1, 53, 8, 291]
    #把每个单词编号映射成一个向量\例如：
	#•	dog -> [0.2, -0.7, 1.1, ...]
	#•	cat -> [0.3, -0.6, 1.0, ...]
    # 为什么一定要变成向量？
    # 向量不是“编号本身”，而是模型学出来的特征表示。
    # 怎么学的？
    # 先随机初始化，然后梯度下降。
    #训练很多轮以后，会发生这种事：
    #经常出现在相似上下文里的词，它们的向量会慢慢变得接近，起相似语法作用的词，也会形成某些规律
    #某些维度会隐式编码一些信息，比如名词/动词、颜色、动物、动作等
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    out = None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using Pytorch's array indexing.         #
    ##############################################################################
    out = W[x]
    #W.shape = (V, D)
    #x.shape = (N, T)
    #W[x]PyTorch 会把 x 里每个元素都当成行索引，从 W 里取对应那一行
    #输出:(N, T, D)
    """import torch

    W = torch.tensor([
        [1.0, 1.1],   # word 0
        [2.0, 2.1],   # word 1
        [3.0, 3.1],   # word 2
        [4.0, 4.1],   # word 3
    ])
    
    x = torch.tensor([
        [0, 2],
        [3, 1]
    ])
    out = [
      [[1.0, 1.1], [3.0, 3.1]],
      [[4.0, 4.1], [2.0, 2.1]]
    ](2,2,2)
    """
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
