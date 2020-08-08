import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import ReLU,BatchNormalization,LayerNormalization,Dense,Input,Lambda
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt 


def createDatasetOH(n):
    X_train=np.repeat(np.eye(256),n,axis=0)
    y_train=np.repeat(np.eye(256),n,axis=0)

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
    return tf.dtypes.cast(count_diff,tf.float32)/(8.0*y_true.shape[0])

class AWGN(tf.keras.layers.Layer):
    def __init__(self,snr,**kwargs):
        super(AWGN,self).__init__(**kwargs)
        self.snr=snr
        self.sigma=np.sqrt(0.5)*10**(-snr/20)
    def call(self,input,training=None):
        if training:
            return tf.identity(input)+tf.stop_gradient(tf.random.normal(tf.shape(input),stddev=self.sigma))
        return tf.identity(input)
    def get_config(self):
        config = super(AWGN, self).get_config()
        config['snr']=self.snr
        return config
    
class Mish(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Mish,self).__init__(**kwargs)
    def call(self,input,training=None):
        return input * tf.math.tanh(tf.math.softplus(input))
    def get_config(self):
        config = super(Mish, self).get_config()
        return config
    
class multi_AWGN(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(multi_AWGN,self).__init__(**kwargs)
    def call(self,input,training=None):
        if training:
            snr=2*np.random.rand()
            sigma=np.sqrt(0.5)*10**(-snr/20)
            return tf.identity(input)+tf.random.normal(tf.shape(input),stddev=sigma)
        return tf.identity(input)
    def get_config(self):
        config = super(multi_AWGN, self).get_config()
        return config 
    
class BSC(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BSC,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            mask=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.uint8)
            return tf.dtypes.cast(tf.bitwise.bitwise_xor(tf.dtypes.cast(input,tf.uint8),mask),tf.float32)
        return tf.identity(input)
    def get_config(self):
        config = super(BSC, self).get_config()
        config['p']=self.p
        return config
    
class BSC_OH(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BSC_OH,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            rounded=tf.identity(input)>0.5
            mask=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.uint8)
            return tf.stop_gradient(tf.dtypes.cast(tf.bitwise.bitwise_xor(tf.dtypes.cast(rounded,tf.uint8),mask),tf.float32)-tf.identity(input))+tf.identity(input)
        return tf.identity(input)
    def get_config(self):
        config = super(BSC_OH, self).get_config()
        config['p']=self.p
        return config
    
class BSC_OH2(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BSC_OH2,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            rounded=tf.math.round(input)
            mask=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.float32)
            output=rounded+mask
            return tf.stop_gradient(tf.identity(output)*(1-tf.dtypes.cast(output>1.5,tf.float32))-tf.identity(input))+tf.identity(input)
        return tf.identity(input)
    def get_config(self):
        config = super(BSC_OH2, self).get_config()
        config['p']=self.p
        return config

class BAC(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BAC,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            temp=tf.dtypes.cast(tf.identity(input),tf.uint8)
            mask1=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.07,tf.uint8)
            mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.uint8)
            mask1=tf.dtypes.cast(mask1*temp,tf.uint8)
            mask2=tf.dtypes.cast(mask2*(1-temp),tf.uint8)
            mask=tf.bitwise.bitwise_xor(mask1,mask2)
            return tf.stop_gradient(tf.dtypes.cast(tf.bitwise.bitwise_xor(tf.identity(temp),mask),tf.float32)-tf.identity(input))+tf.identity(input)
        return tf.identity(input)
    def get_config(self):
        config = super(BAC, self).get_config()
        config['p']=self.p
        return config
    
class BAC_OH(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BAC_OH,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            temp=tf.dtypes.cast(tf.identity(input)>0.5,tf.uint8)
            mask1=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.07,tf.uint8)
            mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.uint8)
            mask1=tf.dtypes.cast(mask1*temp,tf.uint8)
            mask2=tf.dtypes.cast(mask2*(1-temp),tf.uint8)
            mask=tf.bitwise.bitwise_xor(mask1,mask2)
            return tf.stop_gradient(tf.dtypes.cast(tf.bitwise.bitwise_xor(tf.identity(temp),mask),tf.float32)-tf.identity(input))+tf.identity(input)
        return tf.identity(input)
    def get_config(self):
        config = super(BAC_OH, self).get_config()
        config['p']=self.p
        return config
    
class BAC_OH2(tf.keras.layers.Layer):
    def __init__(self,p,**kwargs):
        super(BAC_OH2,self).__init__(**kwargs)
        self.p=p
    def call(self,input,training=None):
        if training:
            rounded=tf.math.round(tf.identity(input))
            mask1=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.07,tf.float32)
            mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.float32)
            mask1=mask1*rounded
            mask2=mask2*(1-rounded)
            mask=mask1+mask2
            mask=mask*(1-tf.dtypes.cast(mask>1.5,tf.float32))
            output=rounded+mask
            return tf.stop_gradient(tf.identity(output)*(1-tf.dtypes.cast(output>1.5,tf.float32))-tf.identity(input))+tf.identity(input)
        return tf.identity(input)
    def get_config(self):
        config = super(BAC_OH2, self).get_config()
        config['p']=self.p
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

def OHEncoder(hidden_size,use_BN=False,use_LN=False,act='tanh'):
    inputs=tf.keras.Input(shape=(256,))
    enc=Dense(hidden_size,activation='relu')(inputs)
    if use_BN:
        enc=BatchNormalization()(enc)
    elif use_LN:
        enc=LayerNormalization()(enc)
    enc_output=Dense(16,activation=act)(enc)
    model=tf.keras.Model(inputs,enc_output)
    return model

def OHDecoder(hidden_size,noise_layer,noise_param=None,use_BN=False,use_LN=False):
    inputs=tf.keras.Input(shape=(16,))
    if noise_param is None:
        noise=noise_layer()(inputs)
    else:
        noise=noise_layer(noise_param)(inputs)
    dec=Dense(hidden_size,activation='relu')(noise)
    if use_BN:
        dec=BatchNormalization()(dec)
    elif use_LN:
        dec=LayerNormalization()(dec)
    dec_output=Dense(256,activation='softmax')(dec)
    model=tf.keras.Model(inputs,dec_output)
    return model

def OHAutoencoder(hidden_size1,hidden_size2,noise_layer,noise_param=None,use_BN=True,use_LN=False,lr=0.001):
    inputs=tf.keras.Input(shape=(256,))
    encoder=OHEncoder(hidden_size1,use_BN,use_LN)
    decoder=OHDecoder(hidden_size2,noise_layer,noise_param,use_BN,use_LN)
    autoencoder=tf.keras.Model(inputs,decoder(encoder(inputs)))
    autoencoder.compile(optimizer=Adam(lr),
              loss='categorical_crossentropy',
              metrics=[ber_metric_oh,'acc'])
    return autoencoder,encoder,decoder

def OHEncoder2(hidden_size,use_BN=False,use_LN=False,act='tanh'):
    inputs=tf.keras.Input(shape=(256,))
    enc=Dense(hidden_size,activation='relu')(inputs)
    if use_BN:
        enc=BatchNormalization()(enc)
        enc=Dense(hidden_size,activation='relu')(enc)
        enc=BatchNormalization()(enc)
    elif use_LN:
        enc=LayerNormalization()(enc)
        enc=Dense(hidden_size,activation='relu')(enc)
        enc=LayerNormalization()(enc)
    else:
        enc=Dense(hidden_size,activation='relu')(enc)
    enc_output=Dense(16,activation=act)(enc)
    model=tf.keras.Model(inputs,enc_output)
    return model

def OHDecoder2(hidden_size,use_BN=False,use_LN=False):
    inputs=tf.keras.Input(shape=(16,))
    dec=Dense(hidden_size,activation='relu')(inputs)
    if use_BN:
        dec=BatchNormalization()(dec)
    elif use_LN:
        dec=LayerNormalization()(dec)
    dec_output=Dense(256,activation='softmax')(dec)
    model=tf.keras.Model(inputs,dec_output)
    return model

def OHAutoencoder2(hidden_size1,hidden_size2,noise_layer,noise_param=None,use_BN=True,use_LN=False,lr=0.001):
    inputs=tf.keras.Input(shape=(256,))
    encoder=OHEncoder(hidden_size1,use_BN,use_LN,'sigmoid')
    decoder=OHDecoder2(hidden_size2,use_BN,use_LN)
    if noise_param is None:
        noise=noise_layer()
    else:
        noise=noise_layer(noise_param)
    enc=encoder(inputs)
    noisy=noise(enc)
    outputs=decoder(noisy)
    autoencoder=tf.keras.Model(inputs,outputs)
    autoencoder.compile(optimizer=Adam(lr),
              loss='categorical_crossentropy',
              metrics=[ber_metric_oh,'acc'])
    return autoencoder,encoder,decoder

def OHEncoder_test(hidden_size,use_BN=False,use_LN=False):
    inputs=tf.keras.Input(shape=(256,))
    enc=Dense(hidden_size)(inputs)
    enc=Mish()(enc)
    if use_BN:
        enc=BatchNormalization()(enc)
    elif use_LN:
        enc=LayerNormalization()(enc)
    enc_output=Dense(16)(enc)
    enc_output=Lambda(lambda x : tf.tanh(2/3*x))(enc_output)
    model=tf.keras.Model(inputs,enc_output)
    return model

def OHDecoder_test(hidden_size,noise_layer,noise_param=None,use_BN=False,use_LN=False):
    inputs=tf.keras.Input(shape=(16,))
    if noise_param is None:
        noise=noise_layer()(inputs)
    else:
        noise=noise_layer(noise_param)(inputs)
    dec=Dense(hidden_size)(noise)
    dec=Mish()(dec)
    if use_BN:
        dec=BatchNormalization()(dec)
    elif use_LN:
        dec=LayerNormalization()(dec)
    dec_output=Dense(256,activation='softmax')(dec)
    model=tf.keras.Model(inputs,dec_output)
    return model

def OHAutoencoder_test(hidden_size1,hidden_size2,noise_layer,noise_param=None,use_BN=True,use_LN=False,lr=0.001):
    inputs=tf.keras.Input(shape=(256,))
    encoder=OHEncoder_test(hidden_size1,use_BN,use_LN)
    decoder=OHDecoder_test(hidden_size2,noise_layer,noise_param,use_BN,use_LN)
    autoencoder=tf.keras.Model(inputs,decoder(encoder(inputs)))
    autoencoder.compile(optimizer=Adam(lr),
              loss='categorical_crossentropy',
              metrics=[ber_metric_oh,'acc'])
    return autoencoder,encoder,decoder

def plot_history(history,to_plot=None,save=True,base_name=""): #if to_plot is None, plot everything, else only plot values in to_plot
    if to_plot is None:
        to_plot=history.keys()
    for metric in to_plot:
        plt.figure()
        plt.title(metric + " vs epoch")
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.plot(np.arange(len(history.history[metric]))+1,history.history[metric])
        if save:
            plt.savefig(base_name+metric+'.eps')
    plt.show()