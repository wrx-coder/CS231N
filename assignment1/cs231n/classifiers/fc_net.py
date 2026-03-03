from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,#初始化权重时高斯分布的标准差（std）
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}#存参数
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        #初始化
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        # affine→ReLU→affine→softmax
        #思路：
        #1.forward 计算
        #2.Test mode
        #3.Training mode
        #data loss dscores
        #加L2正则损失
        #back prop ---dw,dx,dhidden……
        #加正则损失
        #存
        scores = None
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #取参数
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        #前向传播
        hidden, cache_hidden = affine_relu_forward(X, W1, b1)
        scores, cache_scores = affine_forward(hidden, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        dhidden, dW2, db2 = affine_backward(dscores, cache_scores)
        dX, dW1, db1 = affine_relu_backward(dhidden, cache_hidden)

        dW2 += self.reg * W2
        dW1 += self.reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,#一个列表、表示隐藏层的宽度
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,#全部保留，无dropout
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,#float32 更快，float64更精确
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)#隐藏层+一个输出层
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #构造所有层维度列表 
        # Build a list of layer dimensions: [input_dim, h1, h2, ..., hL-1, num_classes]
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        # Initialize weights and biases for all layers
        for i in range(self.num_layers):#隐藏层和输出层L-1+1个w
            W_key = f"W{i+1}"#生成参数名
            b_key = f"b{i+1}"#生成参数名

            self.params[W_key] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i+1])
            self.params[b_key] = np.zeros(layer_dims[i+1])

            # Initialize normalization params (only for hidden layers, not the last output layer)
            if self.normalization in ["batchnorm", "layernorm"] and i < self.num_layers - 1:# 不给最后一层初始化
                gamma_key = f"gamma{i+1}"
                beta_key = f"beta{i+1}"
                self.params[gamma_key] = np.ones(layer_dims[i+1])
                self.params[beta_key] = np.zeros(layer_dims[i+1])
            
                self.params[gamma_key] = np.ones(layer_dims[i+1])#init 缩放参数
                self.params[beta_key] = np.zeros(layer_dims[i+1])#init 平移参数


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        #保存dropout配置
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        #保存归一化的配置
        #BatchNorm 是“按 batch 归一化”，LayerNorm 是“按单个样本内部的特征维度归一化”。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]#输出层不用
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.所有数据类型统一
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #前向传播，记得存cache
        caches = {}
        out = X

        # hidden layers: 1 ... L-1
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            
            out, fc_cache = affine_forward(out, W, b)
            #####################################################
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                out, norm_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i - 1])
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                out, norm_cache = layernorm_forward(out, gamma, beta, self.bn_params[i - 1])
            else:
                norm_cache = None
            ######################################################
            out, relu_cache = relu_forward(out)

            if self.normalization is None:
                caches[i] = (fc_cache, relu_cache)
            else:
                caches[i] = (fc_cache, norm_cache, relu_cache)

        # final layer: L (affine only)
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        scores, final_cache = affine_forward(out, W_last, b_last)
        caches[self.num_layers] = final_cache
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Softmax data loss softmax在最后一层
        loss, dscores = softmax_loss(scores, y)

# Add L2 regularization to loss (weights only)
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)
        
        # Backward pass for final affine layer
        dout, dW, db = affine_backward(dscores, caches[self.num_layers])
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db
        
        # Backward pass through hidden layers in reverse order
        # hidden layers backward: i = self.num_layers-2, ..., 0
        for i in reversed(range(1,self.num_layers)):
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches[i])
        
            if self.normalization is None:
                # cache format: (fc_cache, relu_cache)
                fc_cache, relu_cache = caches[i]
        
                dout = relu_backward(dout, relu_cache)
                dout, dW, db = affine_backward(dout, fc_cache)
        
            else:
                # cache format: (fc_cache, norm_cache, relu_cache)
                fc_cache, norm_cache, relu_cache = caches[i]
        
                dout = relu_backward(dout, relu_cache)
        
                if self.normalization == "batchnorm":
                    dout, dgamma, dbeta = batchnorm_backward_alt(dout, norm_cache)
                elif self.normalization == "layernorm":
                    dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
        
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
        
                dout, dW, db = affine_backward(dout, fc_cache)
        
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True