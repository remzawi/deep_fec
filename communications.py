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

def AWGN_round(x):
    p=(x>=0).astype(np.float32)
    m=(x<0).astype(np.float32)
    return p-m

def correct(x,noise,active=True):
    if not active:
        return x
    if noise=='AWGN':
        return AWGN_round(x)
    else:
        return np.round(x)

def createBitVectors():
    return oh2bin(np.eye(256))

def createPolarCodewords():
    return np.mod(np.dot(createBitVectors(),G),2).astype('float32')

def BPSK(x):
    return 2 * x - 1

def oh2bin(x):
    if x.ndim==1:
        return unpackbits(np.argmax(x),8)
    return unpackbits(np.argmax(x,axis=1),8)

def bin2oh(x):
    dist=np.sum(np.abs(createBitVectors()-x),axis=1)
    return np.eye(256)[np.argmax(dist)]

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

def count_diff(y_result,y_true,one_hot=False,to_int=False):
    if to_int:
        if one_hot:
            y_result=np.eye(256)[np.argmax(y_result,axis=1)]
        else:
            y_result=np.round(y_result)
    if one_hot:
        y_result=oh2bin(y_result)
    return np.count_nonzero(y_result-y_true)

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

def awgn(X_train,snr=1,q=None,with_BPSK=True,noise_only=False,noise_shape=None,apply_noise=None):
    if apply_noise is not None:
        return X_train+apply_noise
    sigma =np.sqrt(0.5)*10**(-snr/20)
    if noise_only:
        return np.random.normal(0,sigma,noise_shape)
    if with_BPSK:
        X_train=BPSK(X_train)
    X_train=X_train+np.random.normal(0,sigma,X_train.shape)
    return X_train

def bsc(X_train,p=0.2,q=None,noise_only=False,noise_shape=None,apply_noise=None):
    if apply_noise is not None:
        return np.mod(X_train+apply_noise,2)
    if noise_only:
        mask=np.random.rand(noise_shape[0],noise_shape[1])<=p
        return mask
    mask=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    return np.mod(X_train+mask,2)

def bac(X_train,p=0.2,q=0.07,noise_only=False,noise_shape=None,apply_noise=None):
    if apply_noise is not None:
        mask1=apply_noise[0]*(1-X_train)
        mask2=apply_noise[1]*X_train
        return np.mod(X_train+mask1+mask2,2)
    if noise_only:
        mask1=np.random.rand(noise_shape[0],noise_shape[1])<=p
        mask2=np.random.rand(noise_shape[0],noise_shape[1])<=q
        return (mask1,mask2)
    mask1=np.random.rand(X_train.shape[0],X_train.shape[1])<=p
    mask2=np.random.rand(X_train.shape[0],X_train.shape[1])<=q
    mask1=mask1*(1-X_train)
    mask2=mask2*X_train
    
    return np.mod(X_train+mask1+mask2,2)



def MAP_AWGN(db,x,p=None,q=None):
    dist=np.sum((db-x)**2,axis=1)
    return unpackbits(np.argmin(dist),8)

def MAP_BSC(db,x,p=None,q=None):
    dist=np.count_nonzero(db-x,axis=1)
    return unpackbits(np.argmin(dist),8)

def MAP_BAC(db,x,p,q=0.07):
    dist=np.log(1-p)*np.count_nonzero(1-db,axis=1)+np.log(1-q)*np.count_nonzero(db,axis=1)
    dist+=np.log(p/(1-p))*np.sum((db<0.5)*np.mod(db+x,2),axis=1)+np.log(q/(1-q))*np.sum((db>0.5)*np.mod(db+x,2),axis=1)
    return unpackbits(np.argmax(dist),8)

def apply_MAP(db,x,MAP,p,q=0.07):
    result=np.zeros((len(x),8))
    for i in range(len(x)):
        result[i]=MAP(db,x[i],p,q)
    return result
                  
def multBER(encoder_list,decoder_list,noise_type,param_values,q_param=0.07,do_polar_MAP=False,enc_MAP_ind=[],save_params=True,params_name=None,save_ber=True,ber_name=None,plot_ber=True,points_per_value=[10**5],plot_legend=None,save_fig=False,fig_name=None):
    n=len(encoder_list)
    m=len(param_values)
    n_map=len(enc_MAP_ind)
    if n != len(decoder_list):
        raise ValueError('missing encoder or decoder')
    if noise_type == 'AWGN':
        noise=awgn
        MAP=MAP_AWGN
    elif noise_type == 'BAC':
        noise=bac
        MAP=MAP_BAC
    elif noise_type == 'BSC':
        noise=bsc
        MAP=MAP_BSC
    else:
        raise ValueError('noise_type not one of (AWGN, BSC, BAC)')
    if len(points_per_value)==1:
        points_per_value=points_per_value*m
    if do_polar_MAP and n_map>0:
        ber_list=np.zeros((n+1+n_map,m))
        db_polar=createPolarCodewords()
        db_enc=[correct(encoder_list[ind].predict(np.eye(256)),noise_type) for ind in enc_MAP_ind]
        if noise_type=='AWGN':
            db_polar=2*db_polar-1
    elif do_polar_MAP:
        ber_list=np.zeros((n+1,m))
        db_polar=createPolarCodewords()
        if noise_type=='AWGN':
            db_polar=2*db_polar-1
    elif n_map>0:
        ber_list=np.zeros((n+n_map,m))
        db_enc=[correct(encoder_list[ind].predict(np.eye(256)),noise_type)for ind in enc_MAP_ind]
    else:
        ber_list=np.zeros((n,m))
    for i in tqdm(range(m)):
        n_pass=points_per_value[i]//10**5
        for k in range(n_pass):        
            ind=np.random.randint(256,size=10**5)
            y_ber=np.zeros((10**5,256))
            y_ber[np.arange(10**5),ind]=1
            y_ber_bin=oh2bin(y_ber)
            noise_sample=noise(None,param_values[i],q=q_param,noise_only=True,noise_shape=(10**5,16))
            for j in range(n):
                X_ber=correct(encoder_list[j].predict(y_ber),noise_type)
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=decoder_list[j].predict(X_ber)
                ber_list[j,i]+=count_diff(y_pred,y_ber_bin,True,True)
            if do_polar_MAP and n_map>0:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
                for l in range(n_map):
                    X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l+1,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif do_polar_MAP:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif n_map>0:
                for l in range(n_map):
                    X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l,i]+=count_diff(y_pred,y_ber_bin,False,False)
        r=points_per_value[i]%10**5
        if r !=0:
            ind=np.random.randint(256,size=r)
            y_ber=np.zeros((r,256))
            y_ber[np.arange(r),ind]=1
            y_ber_bin=oh2bin(y_ber)
            noise_sample=noise(None,param_values[i],q=q_param,noise_only=True,noise_shape=(r,16))
            for j in range(n):
                X_ber=correct(encoder_list[j].predict(y_ber),noise_type)
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=decoder_list[j].predict(X_ber)
                ber_list[j,i]+=count_diff(y_pred,y_ber_bin,True,True)
            if do_polar_MAP and n_map>0:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
                for l in range(n_map):
                    X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l+1,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif do_polar_MAP:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif n_map>0:
                for l in range(n_map):
                    X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l,i]+=count_diff(y_pred,y_ber_bin,False,False)
        ber_list[:,i]/=points_per_value[i]*8
    if save_params:
        if params_name is None:
            params_name='params.npy'
            np.save(params_name,param_values)
        elif len(params_name)==1:
            np.save(params_name[0],param_values)
        elif len(params_name)==len(ber_list):
            for i in range(len(ber_list)):
                np.save(params_name[i],param_values)
        else:
            print("Incorrect parameters name, saving to params.py")
            params_name='params.npy'
            np.save(params_name,param_values)
    if save_ber:
        if ber_name is None:
            ber_name='ber.npy'
            np.save(ber_name,ber_list)
        elif len(ber_name)==1:
            np.save(ber_name[0],ber_list)
        elif len(ber_name)==len(ber_list):
            for i in range(len(ber_name)):
                np.save(ber_name[i],ber_list[i])
        else:
            print("Incorrect ber name, saving to ber.py")
            ber_name='ber.py'
            np.save(ber_name,ber_list)
    if plot_ber:
        plt.figure()
        plt.ylabel('BER')
        plt.xlabel('Parameter')
        plt.yscale('log')
        for i in range(len(ber_list)):
            plt.plot(param_values,ber_list[i])
        if plot_legend is not None:
            plt.legend(plot_legend)
        if save_fig:
            if fig_name is not None:
                plt.savefig(fig_name)
            else:
                plt.savefig('BERvsSNR.pdf')
        plt.show()


def computeBER(encoder_list, decoder_list, enc_types, dec_types, noise_type, param_values, q_param=0.07, do_polar_MAP=False, enc_MAP_ind=[], save_params=True, params_name=None, save_ber=True, ber_name=None, plot_ber=True, points_per_value=[10**5], plot_legend=None, save_fig=False, fig_name=None,correct_output=False):
    n=len(encoder_list)
    m=len(param_values)
    n_map=len(enc_MAP_ind)
    if n != len(decoder_list):
        raise ValueError('missing encoder or decoder')
    if noise_type == 'AWGN':
        noise=awgn
        MAP=MAP_AWGN
    elif noise_type == 'BAC':
        noise=bac
        MAP=MAP_BAC
    elif noise_type == 'BSC':
        noise=bsc
        MAP=MAP_BSC
    else:
        raise ValueError('noise_type not one of (AWGN, BSC, BAC)')
    if len(points_per_value)==1:
        points_per_value=points_per_value*m
    if do_polar_MAP and n_map>0:
        ber_list=np.zeros((n+1+n_map,m))
        db_polar=createPolarCodewords()
        db_enc=[]
        vectors=createBitVectors()
        for ind in enc_MAP_ind:
            if enc_types[ind]=='onehot':
                db_enc.append(correct(encoder_list[ind].predict(np.eye(256)),noise_type,correct_output))
            else:
                db_enc.append(correct(encoder_list[ind].predict(vectors),noise_type,correct_output))
        if noise_type=='AWGN':
            db_polar=2*db_polar-1
    elif do_polar_MAP:
        ber_list=np.zeros((n+1,m))
        db_polar=createPolarCodewords()
        if noise_type=='AWGN':
            db_polar=2*db_polar-1
    elif n_map>0:
        ber_list=np.zeros((n+n_map,m))
        db_enc=[]
        vectors=createBitVectors()
        for ind in enc_MAP_ind:
            if enc_types[ind]=='onehot':
                db_enc.append(correct(encoder_list[ind].predict(np.eye(256)),noise_type,correct_output))
            else:
                db_enc.append(correct(encoder_list[ind].predict(vectors),noise_type,correct_output))
    else:
        ber_list=np.zeros((n,m))
    for i in tqdm(range(m)):
        n_pass=points_per_value[i]//10**5
        for k in range(n_pass):        
            ind=np.random.randint(256,size=10**5)
            y_ber=np.zeros((10**5,256))
            y_ber[np.arange(10**5),ind]=1
            y_ber_bin=oh2bin(y_ber)
            noise_sample=noise(None,param_values[i],q=q_param,noise_only=True,noise_shape=(10**5,16))
            for j in range(n):
                if enc_types[j]=='onehot':
                    X_ber=correct(encoder_list[j].predict(y_ber),noise_type,correct_output)
                else:
                    X_ber=correct(encoder_list[j].predict(y_ber_bin),noise_type,correct_output)
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=decoder_list[j].predict(X_ber)
                ber_list[j,i]+=count_diff(y_pred,y_ber_bin,dec_types[j]=='onehot',True)
            if do_polar_MAP and n_map>0:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
                for l in range(n_map):
                    if enc_types[enc_MAP_ind[l]]=='onehot':
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type,correct_output)
                    else:
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber_bin),noise_type,correct_output)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l+1,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif do_polar_MAP:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif n_map>0:
                for l in range(n_map):
                    if enc_types[enc_MAP_ind[l]]=='onehot':
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type,correct_output)
                    else:
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber_bin),noise_type,correct_output)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l,i]+=count_diff(y_pred,y_ber_bin,False,False)
        r=points_per_value[i]%10**5
        if r !=0:
            ind=np.random.randint(256,size=r)
            y_ber=np.zeros((r,256))
            y_ber[np.arange(r),ind]=1
            y_ber_bin=oh2bin(y_ber)
            noise_sample=noise(None,param_values[i],q=q_param,noise_only=True,noise_shape=(r,16))
            for j in range(n):
                if enc_types[j]=='onehot':
                    X_ber=correct(encoder_list[j].predict(y_ber),noise_type,correct_output)
                else:
                    X_ber=correct(encoder_list[j].predict(y_ber_bin),noise_type,correct_output)
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=decoder_list[j].predict(X_ber)
                ber_list[j,i]+=count_diff(y_pred,y_ber_bin,dec_types[j]=='onehot',True)
            if do_polar_MAP and n_map>0:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
                for l in range(n_map):
                    if enc_types[enc_MAP_ind[l]]=='onehot':
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type,correct_output)
                    else:
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber_bin),noise_type,correct_output)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l+1,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif do_polar_MAP:
                X_ber=np.mod(np.dot(y_ber_bin,G),2)
                if noise_type=='AWGN':
                    X_ber=2.0*X_ber-1
                X_ber=noise(X_ber,apply_noise=noise_sample)
                y_pred=apply_MAP(db_polar,X_ber,MAP,param_values[i],q_param)
                ber_list[n,i]+=count_diff(y_pred,y_ber_bin,False,False)
            elif n_map>0:
                for l in range(n_map):
                    if enc_types[enc_MAP_ind[l]]=='onehot':
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber),noise_type,correct_output)
                    else:
                        X_ber=correct(encoder_list[enc_MAP_ind[l]].predict(y_ber_bin),noise_type,correct_output)
                    X_ber=noise(X_ber,apply_noise=noise_sample)
                    y_pred=apply_MAP(db_enc[l],X_ber,MAP,param_values[i],q_param)
                    ber_list[n+l,i]+=count_diff(y_pred,y_ber_bin,False,False)
        ber_list[:,i]/=points_per_value[i]*8
    if save_params:
        if params_name is None:
            params_name='params.npy'
            np.save(params_name,param_values)
        elif len(params_name)==1:
            np.save(params_name[0],param_values)
        elif len(params_name)==len(ber_list):
            for i in range(len(ber_list)):
                np.save(params_name[i],param_values)
        else:
            print("Incorrect parameters name, saving to params.py")
            params_name='params.npy'
            np.save(params_name,param_values)
    if save_ber:
        if ber_name is None:
            ber_name='ber.npy'
            np.save(ber_name,ber_list)
        elif len(ber_name)==1:
            np.save(ber_name[0],ber_list)
        elif len(ber_name)==len(ber_list):
            for i in range(len(ber_name)):
                np.save(ber_name[i],ber_list[i])
        else:
            print("Incorrect ber name, saving to ber.py")
            ber_name='ber.py'
            np.save(ber_name,ber_list)
    if plot_ber:
        plt.figure()
        plt.ylabel('BER')
        plt.xlabel('Parameter')
        plt.yscale('log')
        for i in range(len(ber_list)):
            plt.plot(param_values,ber_list[i])
        if plot_legend is not None:
            plt.legend(plot_legend)
        if save_fig:
            if fig_name is not None:
                plt.savefig(fig_name)
            else:
                plt.savefig('BERvsSNR.pdf')
        plt.show()
        
def computeVarianceAWGN(encoder,do_polar=True,print_code=False,round=True):
    u=np.eye(256)
    x=encoder.predict(u)
    if round:
        x=AWGN_round(x)
    m=np.mean(x)
    v=np.var(x)
    print('Mean of the code: ',m)
    print('Variance of the code: ',v)
    if do_polar:
        u_polar=oh2bin(u)
        x_polar=2*np.mod(np.dot(u_polar,G),2)-1
        m_polar=np.mean(x_polar)
        v_polar=np.var(x_polar)
        print('Mean of the polar code: ',m_polar)
        print('Variance of the polar code: ',v_polar)
    if print_code:
        print(x)
    
def testLinearAWGN(encoder,round=True):
    ind=np.random.randint(256,size=2)
    u0=unpackbits(ind[0],8)
    u1=unpackbits(ind[1],8)
    u2=np.mod(u0+u1,2)
    print(u0)
    print(u1)
    print(u2)
    u=np.zeros((3,256))
    u[0]=bin2oh(u0)
    u[1]=bin2oh(u1)
    u[2]=bin2oh(u2)
    x=encoder.predict(u)
    if round:
        x=np.round(x)
    x=(x+1)/2
    print(x[0])
    print(x[1])
    print(x[2])
    if np.sum(np.abs(x[2]-np.mod(x[0]+x[1],2)))>0.1:
        print('The code is not linear')
    else:
        print('The code is linear')
    
