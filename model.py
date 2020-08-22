import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import ReLU,BatchNormalization,LayerNormalization,Dense,Input,Lambda
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
#tfa.options.TF_ADDONS_PY_OPS = True

def get_centralized_gradients(optimizer, loss, params):
    """Compute a list of centralized gradients.
    
    Modified version of tf.keras.optimizers.Optimizer.get_gradients:
    https://github.com/keras-team/keras/blob/1931e2186843ad3ca2507a3b16cb09a7a3db5285/keras/optimizers.py#L88-L101
    Reference:
        https://arxiv.org/pdf/2004.01461.pdf
    """
    grads = []
    for grad in K.gradients(loss, params):
        rank = len(grad.shape)
        if rank > 1:
            grad -= tf.reduce_mean(grad, axis=list(range(rank-1)), keep_dims=True)
        grads.append(grad)
    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads

def get_centralized_gradients_function(optimizer):
    """Produce a get_centralized_gradients function for a particular optimizer instance."""

    def get_centralized_gradients_for_instance(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_instance

def createDatasetOH(n):
    X_train=np.repeat(np.eye(256),n,axis=0)
    y_train=np.repeat(np.eye(256),n,axis=0)
    return X_train,y_train

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
            plus=tf.dtypes.cast(tf.identity(input)>=0,tf.float32)
            minus=tf.dtypes.cast(tf.identity(input)<0,tf.float32)
            rounded=plus-minus
            return tf.identity(input)+tf.stop_gradient(rounded+tf.random.normal(tf.shape(input),stddev=self.sigma)-tf.identity(input))
        else:
            plus=tf.dtypes.cast(tf.identity(input)>=0,tf.float32)
            minus=tf.dtypes.cast(tf.identity(input)<0,tf.float32)
            rounded=plus-minus
            return rounded
    def get_config(self):
        config = super(AWGN, self).get_config()
        config['snr']=self.snr
        return config
def AWGN_round(x):
    return x>=0-x<0
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
            if self.p < 1:
                mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<self.p,tf.float32)
            else:
                r=0.35*tf.random.uniform((tf.shape(input)[0],1))
                mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<r,tf.float32)
                #if r[0,0]<1/3:
                #    print('0')
                #    mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.01,tf.float32)
                #elif r[0,0]<2/3:
                #    print('1')
                #    mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.15,tf.float32)
                #else:
                #    print('2')
                #    mask2=tf.dtypes.cast(tf.random.uniform(tf.shape(input))<0.3,tf.float32)
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

def OHEncoder(hidden_size,use_BN=False,use_LN=False,encoder_activation='tanh',hidden_activation='relu'):
    inputs=tf.keras.Input(shape=(256,))
    enc=Dense(hidden_size,activation=hidden_activation)(inputs)
    if use_BN:
        enc=BatchNormalization()(enc)
    elif use_LN:
        enc=LayerNormalization()(enc)
    enc_output=Dense(16,activation=encoder_activation)(enc)
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

def OHDecoder2(hidden_size,use_BN=False,use_LN=False,hidden_activation='relu'):
    inputs=tf.keras.Input(shape=(16,))
    dec=Dense(hidden_size,activation=hidden_activation)(inputs)
    if use_BN:
        dec=BatchNormalization()(dec)
    elif use_LN:
        dec=LayerNormalization()(dec)
    dec_output=Dense(256,activation='softmax')(dec)
    model=tf.keras.Model(inputs,dec_output)
    return model

def OHAutoencoder2(hidden_size1,hidden_size2,noise_layer,noise_param=None,use_BN=True,use_LN=False,lr=0.001,encoder_activation='sigmoid',hidden_activation='relu',optim=Adam,lookahead=False,gradient_centralization=False):
    inputs=tf.keras.Input(shape=(256,))
    encoder=OHEncoder(hidden_size1,use_BN,use_LN,encoder_activation,hidden_activation)
    decoder=OHDecoder2(hidden_size2,use_BN,use_LN,hidden_activation)
    if noise_param is None:
        noise=noise_layer()
    else:
        noise=noise_layer(noise_param)
    enc=encoder(inputs)
    noisy=noise(enc)
    outputs=decoder(noisy)
    autoencoder=tf.keras.Model(inputs,outputs)
    
    if lookahead:
        optim=tfa.optimizers.Lookahead(optim(lr))
    else:
        optim=optim(lr)
    if gradient_centralization:
        optim.get_gradients=get_centralized_gradients_function(optim)
    autoencoder.compile(optimizer=optim,
               loss='categorical_crossentropy',
               metrics=[ber_metric_oh,'acc'])
    return autoencoder,encoder,decoder

def recompile(encoder,decoder,noise_layer,noise_param=None,optim=Adam,lookahead=False,gradient_centralization=False,lr=0.001):
    inputs=tf.keras.Input(shape=(256,))
    if noise_param is None:
        noise=noise_layer()
    else:
        noise=noise_layer(noise_param)
    enc=encoder(inputs)
    noisy=noise(enc)
    outputs=decoder(noisy)
    autoencoder=tf.keras.Model(inputs,outputs)
    
    if lookahead:
        optim=tfa.optimizers.Lookahead(optim(lr))
    if gradient_centralization:
        optim.get_gradients=get_centralized_gradients_function(optim)
    autoencoder.compile(optimizer=optim,
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
    enc_output=Dense(16,activation='sigmoid')(enc)
    model=tf.keras.Model(inputs,enc_output)
    return model

def OHDecoder_test(hidden_size,noise_layer,noise_param=None,use_BN=False,use_LN=False):
    inputs=tf.keras.Input(shape=(16,))
    dec=Dense(hidden_size)(inputs)
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
    decoder=OHDecoder_test(hidden_size2,use_BN,use_LN)
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

