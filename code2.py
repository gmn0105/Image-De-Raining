import tensorflow as tf

alpha=0
def prelu(x):
    global alpha
    alphas = tf.get_variable(str(alpha), x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    alpha+=1
    return pos + neg


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ConvLayer:
    def __init__(self, inp, inpK, K, strideX, strideY, pad, filterSize, BN, PRelu):
        # self.out = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
        # if(BN == True):
        #     self.out = tf.layers.batch_normalization(conv)
        # if(PRelu == True):
        #     self.out = prelu(self.out)
        # else:
        #     self.out = tf.nn.relu(self.out)
        self.inp = inp
        # self.x=[int(inp.shape[0])]
        self.strideX = strideX
        self.strideY = strideY
        self.pad = pad
        self.K = K
        self.inpK = inpK
        self.filterW = weight_variable([filterSize, filterSize, inpK, K])
        self.bias = bias_variable([self.K])
        self.deconv_filterW = weight_variable([filterSize, filterSize, K, inpK])
        self.BN = BN
        self.PRelu = PRelu

    def forward(self):
        out = tf.nn.conv2d(self.inp, self.filterW, strides=[1, self.strideX, self.strideY, 1], padding='SAME') + self.bias
        if (self.BN == True):
            out = tf.layers.batch_normalization(out)
        if (self.PRelu == True):
            out = prelu(out)
        else:
            out = tf.nn.relu(out)
        return out

    def deconv_forward(self):
        out = tf.nn.conv2d_transpose(self.inp, self.deconv_filterW, output_shape=tf.constant([7,256,256,self.K]),strides=[1, self.strideX, self.strideY, 1], padding='SAME') + self.bias
        if (self.BN == True):
            out = tf.layers.batch_normalization(out)
        if (self.PRelu == True):
            out = prelu(out)
        else:
            out = tf.nn.relu(out)
        return out

class DeepNet:
    def __init__(self,inp):
        # self.inp = tf.flatten(inp)
        self.inp = tf.reshape(inp,[-1,32*32])
        self.W = weight_variable([32*32,1])
        self.bias = bias_variable([1])

    def forward(self):
        out = tf.matmul(self.inp,self.W) + self.bias
        out = tf.sigmoid(out)
        return out

class Generator:
    def __init__(self):
        self.layers = []

    def addConvLayer(self, inp, inpK, K, strideX=1, strideY=1, pad='SAME', filterSize=3, BN=True, PRelu=True):
        # def __init__(self,inp,inpK,K,strideX=1,strideY=1,pad='SAME',filterSize=5):
        cLayer = ConvLayer(inp, inpK, K, strideX, strideY, pad, filterSize, BN, PRelu)
        self.layers.append(cLayer)
        return cLayer.forward()

    def addDeConvLayer(self, inp, inpK, K, strideX=1, strideY=1, pad='SAME', filterSize=3, BN=True, PRelu=True):
        # def __init__(self,inp,inpK,K,strideX=1,strideY=1,pad='SAME',filterSize=5):
        cLayer = ConvLayer(inp, inpK, K, strideX, strideY, pad, filterSize, BN, PRelu)
        self.layers.append(cLayer)
        return cLayer.deconv_forward()

    def addDeepNet(self,inp):
        cLayer = DeepNet(inp)
        self.layers.append(cLayer)
        return cLayer.forward()

    def setInput(self,inputTensor):
        self.layers[0].inp=inputTensor

    def forward(self,inp):
        out = inp
        self.setInput(inp)
        for layer in self.layers:
            out = layer.forward()
        return out
