import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import ReLU,BatchNormalization,LayerNormalization,Dense

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

class OHDecoder(tf.keras.Model):
    def __init__(self,hidden_size):
        super(Decoder,self).__init__()
        self.hidden=Dense(hidden_size)
        self.relu=ReLU()
        self.ln1=LayerNormalization()
        self.ln2=LayerNormalization()
        self.out=Dense(10,activation='softmax')
    
    def call(self, input_tensor, training=False):
        x=self.ln1(input_tensor)
        x=self.hidden(x)
        x=self.relu(x)
        x=self.ln2(x)
        x=self.out(x)
        return x

def create_model(hidden_size,save_name):
    model=OHDecoder(hidden_size)
    checkpoint=tf.keras.Callbacks.ModelCheckpoint(save_name,monitor='ber_metric_oh',save_best_only=True)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              callbacks=[checkpoint],
              metrics=[ber_metric_oh])

        


