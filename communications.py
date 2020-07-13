import numpy as np
from tqdm import tqdm

#Default encoding matrix
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

def unpackbits(x, num_bits,to_array=False):
  if to_array:
      x=np.array(x)
  xshape = list(x.shape)
  x = x.reshape([-1, 1])
  mask = 2**np.arange(num_bits).reshape([1, num_bits])
  return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def createBitVectors():
    x=np.zeros((256,1),dtype='uint8')
    x[:,0]=np.arange(0,256,dtype='uint8')
    return np.unpackbits(x,axis=1)

def createPolarCodewords():
    return np.mod(np.dot(createBitVectors(),G),2)

def BPSK(x):
    return 2 * x - 1

def oh2bin(x):
    if x.ndim==1:
        return unpackbits(np.argmax(x),8)
    return unpackbits(np.argmax(x,axis=1),8)

def createCdwsDB():
    np.save('polarDB.npy',BPSK(createPolarCodewords()))

def loadDB(BPSK=True):
    if BPSK:
        try:
            return BPSK(np.load('polarDB.npy'))
        except:
            createCdwsDB()
            return BPSK(np.load('polarDB.npy'))
    else:
        try:
            return np.load('polarDB.npy')
        except:
            createCdwsDB()
            return np.load('polarDB.npy')




def BER(y_result,y_true,one_hot=False,to_int=False):
    if to_int:
        if one_hot:
            y_result=np.eye(256)[np.argmax(y_result,axis=1)]
        else:
            y_result=y_result>=0.5
    if one_hot:
        y_result=oh2bin(y_result)
    return np.count_nonzero(y_result-y_true)/np.prod(y_result.shape)

def createTrainingDatasetDecoder(BPSK=True,noise=None,one_hot=False):
    X_train=createPolarCodewords()
    if BPSK:
        X_train=BPSK(X_train)
    if noise is not None:
        X_train=noise(X_train)
    if one_hot:
        return X_train,np.eye(256)
    else:
        return X_train,createBitVectors()

def AWGN(X_train,snr=2):
    sigma =np.sqrt(0.5)*10**(-snr/20)
    X_train+=np.random.normal(0,sigma,X_train.shape)
    return X_train

def BSC(X_train,p=0.2):
    mask=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    return np.mod(X_train+mask,2)

def BAC(X_train,p=0.2,q=0.1):
    mask1=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    mask2=np.random.rand(X_train.shape[0],X_train.shape[1])<=q
    mask1=mask1*(1-X_train)
    mask2=mask2*X_train
    return np.mod(X_train+mask1+mask2,2)



def MAP_AWGN(x):
    db=loadDB()
    dist=np.sum((db-x)**2,axis=1)
    return unpackbits(np.argmin(dist),8)

def MAP_BSC(x):
    db=loadDB(BPSK=False)
    dist=np.count_nonzero(db-x,axis=1)
    return unpackbits(np.argmin(dist),8)

def MAP_BAC(x,p,q):
    dist=np.log(1-p)*np.count_nonzero(1-db,axis=1)+np.log(1-q)*np.count_nonzero(db,axis=1)
    dist+=np.power(np.log(p/(1-p)),np.sum((db<0.5)*x,axis=1))
    dist+=np.power(np.log(q/(1-q)),np.sum((db>0.5)*x,axis=1))