import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    return np.mod(np.dot(createBitVectors(),G),2).astype('float32')

def BPSK(x):
    return 2 * x - 1

def oh2bin(x):
    if x.ndim==1:
        return unpackbits(np.argmax(x),8)
    return unpackbits(np.argmax(x,axis=1),8)

def createCdwsDB():
    np.save('polarDB.npy',createPolarCodewords())

def loadDB(with_BPSK=True):
    if with_BPSK:
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
        X_train=2*X_train-1
    if noise is not None:
        X_train=noise(X_train)
    if one_hot:
        return X_train,np.eye(256)
    else:
        return X_train,createBitVectors()

def AWGN(X_train,snr=1,with_BPSK=True):
    sigma =np.sqrt(0.5)*10**(-snr/20)
    if with_BPSK:
        X_train=BPSK(X_train)
    X_train=X_train+np.random.normal(0,sigma,X_train.shape)
    return X_train

def BSC(X_train,p=0.2):
    mask=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    return np.mod(X_train+mask,2)

def BAC(X_train,p=0.2,q=0.07):
    mask1=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    mask2=np.random.rand(X_train.shape[0],X_train.shape[1])<=q
    mask1=mask1*(1-X_train)
    mask2=mask2*X_train
    return np.mod(X_train+mask1+mask2,2)



def MAP_AWGN(x):
    db=loadDB()
    dist=np.sum((db-x)**2,axis=1)
    return unpackbits(np.argmin(dist),8)

def apply_MAP_AWGN(x,flip=False):
    if flip:
        return np.flip(np.array([MAP_AWGN(x[i]) for i in range(len(x))]),axis=1)
    return np.array([MAP_AWGN(x[i]) for i in range(len(x))])

def MAP_BSC(x):
    db=loadDB(with_BPSK=False)
    dist=np.count_nonzero(db-x,axis=1)
    return unpackbits(np.argmin(dist),8)

def MAP_BAC(x,p,q):
    db=loadDB(with_BPSK=False)
    dist=np.log(1-p)*np.count_nonzero(1-db,axis=1)+np.log(1-q)*np.count_nonzero(db,axis=1)
    dist+=np.log(p/(1-p))*np.sum((db<0.5)*x,axis=1)+np.log(q/(1-q))*np.sum((db>0.5)*x,axis=1)
    return unpackbits(np.argmin(dist),8)


def computePOLARBER(f,noise_type,param_values,one_hot=True,save_params=True,params_name=None,save_ber=True,ber_name=None,plot_ber=True,points_per_value=[10**6]):
    if noise_type == 'AWGN':
        noise=AWGN
    elif noise_type == 'BAC':
        noise=BAC
    elif noise_type == 'BSC':
        noise=BSC
    else:
        raise ValueError('noise_type not one of (AWGN, BSC, BAC)')
    if len(points_per_value)==1:
        points_per_value=points_per_value*len(param_values)
    ber_list=[]
    for i in range(len(param_values)):
        ind=np.random.randint(256,size=points_per_value[i])
        y_ber=createBitVectors()[ind]
        X_ber=np.mod(np.dot(y_ber,G),2).astype('float32')
        X_ber=noise(X_ber,param_values[i])
        y_pred=f(X_ber)
        ber_list.append(BER(y_pred,y_ber,one_hot,True))
    if save_params:
        if params_name is None:
            params_name='params.npy'
        np.save(params_name,param_values)
    if save_ber:
        if ber_name is None:
            ber_name='ber.npy'
        np.save(ber_name,ber_list)
    if plot_ber:
        plt.figure()
        plt.ylabel('BER')
        plt.xlabel('Parameter')
        plt.yscale('log')
        plt.plot(param_values,ber_list)
        plt.show()
    
def test_MAP():
    ind=np.random.randint(256,size=10**4)
    y_ber=unpackbits(ind,8)
    X_ber=np.mod(np.dot(y_ber,G),2).astype('float32')
    X_ber=AWGN(X_ber,8)
    y_pred=apply_MAP_AWGN(X_ber)
    print(BER(np.flip(y_pred,axis=1),y_ber,one_hot=False,to_int=True))
    
def OHAutoencoderBER(encoder,decoder,noise_type,param_values,save_params=True,params_name=None,save_ber=True,ber_name=None,plot_ber=True,points_per_value=[10**6]):
    if noise_type == 'AWGN':
        noise=AWGN
    elif noise_type == 'BAC':
        noise=BAC
    elif noise_type == 'BSC':
        noise=BSC
    else:
        raise ValueError('noise_type not one of (AWGN, BSC, BAC)')
    if len(points_per_value)==1:
        points_per_value=points_per_value*len(param_values)
    ber_list=[]
    for i in range(len(param_values)):
        ind=np.random.randint(256,size=points_per_value[i])
        y_ber=np.zeros((points_per_value[i],256))
        y_ber[np.arange(points_per_value[i]),ind]=1
        X_ber=encoder.predict(y_ber)
        X_ber=noise(X_ber,param_values[i])
        y_pred=decoder.predict(X_ber)
        y_ber=oh2bin(y_ber)
        ber_list.append(BER(y_pred,y_ber,True,True))
    if save_params:
        if params_name is None:
            params_name='params.npy'
        np.save(params_name,param_values)
    if save_ber:
        if ber_name is None:
            ber_name='ber.npy'
        np.save(ber_name,ber_list)
    if plot_ber:
        plt.figure()
        plt.ylabel('BER')
        plt.xlabel('Parameter')
        plt.yscale('log')
        plt.plot(param_values,ber_list)
        plt.show()

#test_MAP()

