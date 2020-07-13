import tensorflow as tf 
import numpy as np

def ber_metric(y_true,y_pred):
    y_pred=y_pred>=0.5
    return tf.math.count_nonzero(y_pred-y_true)/tf.math.reduce_prod(y_pred.shape)

def unpackbits(x):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.range(8).reshape([1, 8])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [8])

def ber_metric_oh(y_true,y_pred):
    y_pred=tf.eye(256)[tf.math.argmax(y_pred,axis=1)]
    y_pred=y_pred.numpy()
    y_true=y_true.numpy()
    y_pred=unpackbits(y_pred)
    y_true=unpackbits(y_true)
    return np.count_nonzero(y_pred-y_true)/np.prod(y_pred.shape)




