import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import ReLU,BatchNormalization,LayerNormalization,Dense,Input
import matplotlib.pyplot as plt 


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
        count_diff+=tf.math.count_nonzero(tf.math.abs(unpacked_pred-unpacked_true)>0.1)
    return tf.dtypes.cast(count_diff,tf.float32)/tf.dtypes.cast(tf.math.reduce_prod(y_pred.shape),tf.float32)

class AWGN(tf.keras.layers.Layer):
    def __init__(self,snr,**kwargs):
        super(AWGN,self).__init__(**kwargs)
        self.sigma=np.sqrt(0.5)*10**(-snr/20)
    def call(self,input,training=None):
        if training:
            return input+tf.random.normal(input.shape,stddev=self.sigma)
        return input
    def get_config(self):
        config = super(AWGN, self).get_config()
        return config
    
class BSC(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BSC,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            mask=tf.dtypes.cast(tf.random.uniform(input.shape)<self.p,tf.uint8)
            return tf.dtypes.cast(tf.bitwise.bitwise_xor(tf.dtypes.cast(input,tf.uint8),mask),tf.float32)
        return input
    def get_config(self):
        config = super(BSC, self).get_config()
        return config

class BAC(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BAC,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            temp=tf.dtypes.cast(input,tf.uint8)
            mask1=tf.dtypes.cast(tf.random.uniform(input.shape)<0.07,tf.uint8)
            mask2=tf.dtypes.cast(tf.random.uniform(input.shape)<self.p,tf.uint8)
            mask1=tf.dtypes.cast(mask1*temp,tf.uint8)
            mask2=tf.dtypes.cast(mask2*(1-temp),tf.uint8)
            mask=tf.bitwise.bitwise_xor(mask1,mask2)
            return tf.dtypes.cast(tf.bitwise.bitwise_xor(temp,mask),tf.float32)
        return input
    def get_config(self):
        config = super(BAC, self).get_config()
        return config
        
def OHDecoder_SEQ(hidden_size,noise_layer,noise_param):
    model=tf.keras.Sequential()
    model.add(noise_layer(noise_param))
    model.add(Dense(hidden_size,activation='relu'))
    model.add(LayerNormalization())
    model.add(Dense(256,activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[ber_metric_oh,'acc'])
    return model

def OHEncoder(hidden_size,use_BN,use_LN):
    inputs=tf.keras.Input(shape=(256,))
    enc=Dense(hidden_size,activation='relu')(inputs)
    if use_BN:
        enc=BatchNormalization()(enc)
    elif use_LN:
        enc=LayerNormalization()(enc)
    enc_output=Dense(16,activation='tanh')(enc)
    model=tf.keras.Model(inputs,enc_output)
    return model

def OHDecoder(hidden_size,noise_layer,noise_param,use_BN,use_LN):
    inputs=tf.keras.Input(shape=(16,))
    noise=noise_layer(noise_param)(inputs)
    dec=Dense(hidden_size,activation='relu')(noise)
    if use_BN:
        dec=BatchNormalization()(dec)
    elif use_LN:
        dec=LayerNormalization()(dec)
    dec_output=Dense(256,activation='softmax')(dec)
    model=tf.keras.Model(inputs,dec_output)
    return model

def OHAutoencoder(hidden_size1,hidden_size2,noise_layer,noise_param,use_BN=True,use_LN=False):
    inputs=tf.keras.Input(shape=(256,))
    encoder=OHEncoder(hidden_size1,use_BN,use_LN)
    decoder=OHDecoder(hidden_size2,noise_layer,noise_param,use_BN,use_LN)
    autoencoder=tf.keras.Model(inputs,decoder(encoder(inputs)))
    autoencoder.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[ber_metric_oh,'acc'])
    return autoencoder,encoder,decoder


def plot_history(history,to_plot=None,save=True,base_name=""): #if to_plot is None, plot everything, else only plot values in to_plot
    if to_plot is None:
        to_plot=history.history.keys()
    for metric in to_plot:
        plt.figure()
        plt.title(metric + " vs epoch")
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.plot(np.arange(len(history.history[metric]))+1,history.history[metric])
        if save:
            plt.savefig(base_name+metric+'.eps')
    plt.show()