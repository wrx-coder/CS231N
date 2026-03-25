from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


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
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
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
        self.num_layers = 1 + len(hidden_dims)
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
        
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] = weight_scale * np.random.randn(
                layer_dims[i - 1], layer_dims[i]
            )
            #先从标准正态分布采样，再乘上 weight_scale
            self.params[f"b{i}"] = np.zeros(layer_dims[i])
        
            # 只有隐藏层才需要 normalization 参数，最后一层输出层不需要
            if i < self.num_layers and self.normalization in ["batchnorm", "layernorm"]:
                self.params[f"gamma{i}"] = np.ones(layer_dims[i])
                self.params[f"beta{i}"] = np.zeros(layer_dims[i])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
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
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        #这里的 __init__ 这部分，本质上就是做一件事：
        #把整张网络每一层要学的参数，全部按名字初始化到 self.params 里。

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
        #前向传播:
        #对每个隐藏层：
	#1.	取出 W{i} 和 b{i}
	#2.	如果要 norm，再取 gamma{i} 和 beta{i}
	#3.	做 hidden block 前向
	#4.	把 cache 存起来
	#5.	如果用了 dropout，再做 dropout_forward
	#6.	把 dropout cache 也存起来

        #最后一层：
	#1.	取出 W_last 和 b_last
	#2.	做 affine_forward
	#3.	得到 scores
        
        caches = {}
        dropout_caches = {}
        out = X
        # out 永远表示：当前这一层要处理的输入 或 上一层的输出

    
        for i in range(1, self.num_layers):
            #取参数
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
    
            if self.normalization is None:
                out, caches[i] = affine_relu_forward(out, W, b)
    
            elif self.normalization == "batchnorm":
                #再拿gamma和beta
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                
                out, caches[i] = affine_bn_relu_forward(
                    out, W, b, gamma, beta, self.bn_params[i - 1]
                )
    
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                out, caches[i] = affine_ln_relu_forward(
                    out, W, b, gamma, beta, self.bn_params[i - 1]
                )
    
            if self.use_dropout:
                out, dropout_caches[i] = dropout_forward(out, self.dropout_param)
    
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        scores, caches[self.num_layers] = affine_forward(out, W_last, b_last)
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
        #反向传播
        loss, dout = softmax_loss(scores, y)

        #加正则项
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)
            
    #为什么最后一层单独处理?因为最后一层没有 relu / norm / dropout，只有 affine。
        dout, dW, db = affine_backward(dout, caches[self.num_layers])
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db
    
        for i in range(self.num_layers - 1, 0, -1):#假设4层，3，2，1
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches[i])
    
            if self.normalization is None:
                dout, dW, db = affine_relu_backward(dout, caches[i])
                grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
                grads[f"b{i}"] = db
    
            elif self.normalization == "batchnorm":
                dout, dW, db, dgamma, dbeta = affine_bn_relu_backward(dout, caches[i])
                grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
                grads[f"b{i}"] = db
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
    
            elif self.normalization == "layernorm":
                dout, dW, db, dgamma, dbeta = affine_ln_relu_backward(dout, caches[i])
                grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
                grads[f"b{i}"] = db
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta

    

#先从 softmax_loss 得到 scores 的梯度，
#再先拆最后一层 affine，
#然后按从后往前的顺序，把每个 hidden layer 依次拆开；
#如果用了 dropout，就先过 dropout backward；
#如果用了 norm，就顺带求出 gamma 和 beta 的梯度；
#最后别忘了给每个 W 的梯度加上正则项。
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

