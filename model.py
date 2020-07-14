import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import ReLU,BatchNormalization,LayerNormalization,Dense

def ber_metric(y_true,y_pred):
    y_pred=y_pred>0.5
    return tf.math.count_nonzero(y_pred-y_true)/tf.math.reduce_prod(y_pred.shape)

def unpackbits(x):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.range(8).reshape([1, 8])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [8])

def ber_metric_oh(y_true,y_pred):
    b = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)
    count_diff=0
    for i in range(y_true.shape[0]):      
        unpacked_true = tf.dtypes.cast(tf.reshape(tf.bitwise.bitwise_and(tf.dtypes.cast(tf.math.argmax(y_true[i]),tf.uint8), b), [-1]),tf.float32)
        unpacked_pred = tf.dtypes.cast(tf.reshape(tf.bitwise.bitwise_and(tf.dtypes.cast(tf.math.argmax(y_pred[i]),tf.uint8), b), [-1]),tf.float32)
        count_diff+=tf.math.count_nonzero((unpacked_pred-unpacked_true)>0.1)
    return tf.dtypes.cast(count_diff,tf.float32)/tf.dtypes.cast(tf.math.reduce_prod(y_pred.shape),tf.float32)

class AWGN(tf.keras.layers.Layer):
    def __init__(self,snr,**kwargs):
        super(AWGN,self).__init__(**kwargs)
        self.sigma=np.sqrt(0.5)*10**(-snr/20)
    def __call__(self,input,training=None):
        if training:
            return input+tf.random.normal(input.shape,stddev=self.sigma)
        return input
        

class OHDecoder(tf.keras.Model):
    def __init__(self,hidden_size,snr):
        super(OHDecoder,self).__init__()
        self.noise=AWGN(snr)
        self.hidden=Dense(hidden_size)
        self.relu=ReLU()
        self.ln1=LayerNormalization()
        self.ln2=LayerNormalization()
        self.out=Dense(256,activation='softmax')
    
    def call(self, input_tensor, training=False):
        x=self.noise(input_tensor,training)
        x=self.ln1(x,training=training)
        x=self.hidden(x)
        x=self.relu(x)
        x=self.ln2(x,training=training)
        x=self.out(x)
        return x

def create_model(hidden_size,snr):
    model=OHDecoder(hidden_size,snr)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',metrics=[ber_metric_oh,'acc'])
    return model

        


