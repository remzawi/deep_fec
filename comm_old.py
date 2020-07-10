# Imported modules

import random as rd
import numpy as np
import matplotlib.pyplot as plt

#Parameters of the simulation

SNR=1
k=8
n=16
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

Gh=np.transpose(np.array([[1,1,0,1],
              [1,0,1,1],
              [1,0,0,0],
              [0,1,1,1],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]]))


cdwsDB=np.load('cdwsDBBPSK.npy',allow_pickle=True)

#Params for BER
MAXSNR=5
MINSNR=0
NOFSNR=6 #number of snrs to test, uniformly distributed between 0 and MAXSNR
TESTSIZE=5000 #size of packets for computing parckets
MINERROR=10 #minimum number of errors (for each function) before stopping BER computation

NOISETYPE=1 #1 AWGN, 2 Laplacien

def changeSimulationParameters(newMAXSNR,newNOFSNR,newTESTSIZE,newMINERROR):
    global MAXSNR
    MAXSNR = newMAXSNR
    global NOFSNR
    NOFSNR = newNOFSNR
    global TESTSIZE
    TESTSIZE = newTESTSIZE
    global MINERROR
    MINERROR = newMINERROR

def createAll():
    t=np.zeros(2**k,k)
    for i in range(2**k):
        t[i]=int2bin2(i)
    return t
 
def createBitvector(length=k):
    rd.seed()
    bv=np.random.randint(0,2,length)
    return bv
def kTon(bv):
    
    return np.mod(np.dot(bv,G),2)

def multH(bv):
    return np.mod(np.dot(bv,Gh),2)

def BPSK(bv):
    return 2*bv-1


def addNoise(bv,snr=SNR,noisetype=1):
    bn = bv.copy()
    sigma =np.sqrt(0.5)*10**(-snr/20)
    if bn.ndim >=2:
        if noisetype==1:
            bn= bn +np.random.normal(0,sigma,(len(bn),len(bv[0])))
        else : 
            bn=bn + np.random.laplace(0,sigma,(len(bn),len(bv[0])))
    else :
        if noisetype==1:
            bn= bn +np.random.normal(0,sigma,len(bn))
        else : 
            bn=bn + np.random.laplace(0,sigma,len(bn))
    return bn


def int2bin(n):
    if n > 1 :
        return int2bin(n >> 1) + [n & 1] 
    elif (n == 1) :
        return [1]
    else : return [0]
 
def int2bin2(n,length=k):
    t=int2bin(n)
    return np.array([0 for i in range (length-len(t)) ] + t)

def bin2int(bv):
    out = 0
    for bit in bv.astype('int32'):
        out = (out << 1) | bit
    return out

def bin2OH(bv):
    t=np.zeros(2**k)
    t[bin2int(bv)]=1
    return t

def OH2bin(oh):
    i=0
    for j in range(len(oh)):
        if oh[j]:
            i=j
    return int2bin2(i)

def OHtoCdws(t):
    res=[]
    for i in range(len(t)):
        res.append(OH2bin(t[i]))
    return np.array(res)

def CdwstoOH(t):
    res=[]
    for i in range(len(t)):
        res.append(bin2OH(t[i]))
    return np.array(res)

def createCodewordsDatabase():
    t=[]
    for i in range(2**k):
        t.append([kTon(int2bin2(i)),int2bin2(i)])
    return t

def isIn(bv,t):
    boolean=False
    for i in range(len(t)):
        boolean = boolean or (t[i]==bv).all()
    return boolean

def createTrainingSet(N,noise=False,snr=SNR):
    X,Y=np.zeros((N,n)),np.zeros((N,k))
    t=[]
    i=0
    while i<N :
        bv = createBitvector()
        if ( not isIn(bv,t)):
            if noise:
                X[i],Y[i]=addNoise(BPSK(kTon(bv)),snr),bv
            else :
                X[i],Y[i]=BPSK(kTon(bv)),bv
            i+=1
            t.append(bv)
    return X,Y


def createAllBitvectors(length=k):
    t=[]
    for i in range(2**length):
        t.append(int2bin2(i,length))
    return t 
def calcDistance(cdw1,cdw2):
    return np.sum(np.power(cdw1-cdw2,2))

def calcNorm(v1,v2):
    return np.sum(np.abs(v1-v2))

def MAP(cdw,DB=cdwsDB,length=k):

    mini =  calcDistance(cdw, DB[0][0])
    res=DB[0][1]
    for i in range(1, 2**length):
        c=calcDistance(cdw, DB[i][0])
        if (c < mini):
            mini = c
            res=DB[i][1]
    return res

def MAPL(cdw,DB=cdwsDB,length=k):

    mini =  calcNorm(cdw, DB[0][0])
    res=DB[0][1]
    for i in range(1, 2**length):
        c=calcNorm(cdw, DB[i][0])
        if (c < mini):
            mini = c
            res=DB[i][1]
    return res

def applyMAP(x_train,DB=cdwsDB,length=k):
    res=[]
    for i in range(len(x_train)):
        res.append(MAP(x_train[i],DB,length))
    return res

def applyMAPL(x_train,DB=cdwsDB,length=k):
    res=[]
    for i in range(len(x_train)):
        res.append(MAPL(x_train[i],DB,length))
    return res

def createTestSet(N,noise=True,snr=SNR,noisetype=NOISETYPE):
    X,Y=np.zeros((N,n)),np.zeros((N,k))
    for i in range(N):
         bv=createBitvector()
         if noise :
             X[i],Y[i]=addNoise(BPSK(kTon(bv)),snr,noisetype),bv
         else : 
             X[i],Y[i]=BPSK(kTon(bv)),bv
    return X,Y


def thresoldDecoder(cdw):
    bv=(cdw>=0)  
    return bv.astype('int8')

def applyTD(x_test):
    t=[]
    for i in range(len(x_test)):
        t.append(thresoldDecoder(x_test[i]))
    return t

def BER(y_result,y_test):
    S=0.0
    N=len(y_result)
    for i in range(N):
        S += calcDistance(y_result[i], y_test[i])
    return S/(k*N)


def calcNbrError(y_result,y_test):
    S=0.0
    N=len(y_result)
    for i in range(N):
        S += calcDistance(y_result[i], y_test[i])
    return S

def createTestAE(encode,N,snr=SNR):
    Y=np.zeros((N,k))
    for i in range(N):
        bv=createBitvector()
        Y[i]=bv
    X=addNoise(encode(Y),snr)
    return X,Y

def createTestOH(encode,N,snr=SNR,noisetype=1):
    Y=np.zeros((N,k))
    for i in range(N):
        bv=createBitvector()
        Y[i]=bv
    X=addNoise(encode(CdwstoOH(Y)),snr,noisetype)
    return X,Y

def createTestUBPSK(N,snr=SNR,noisetype=1):
    Y=np.zeros((N,k))
    for i in range(N):
        bv=createBitvector()
        Y[i]=bv
    X=addNoise(BPSK(Y),snr,noisetype)
    return X,Y

def createTestHamming(N,snr=SNR,noisetype=1):
    Y=np.zeros((N,4))
    for i in range(N):
        bv=createBitvector(4)
        Y[i]=bv
    X=addNoise(BPSK(multH(Y)),snr,noisetype)
    return X,Y

def calcBER(f,minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR):
    nf=len(f)
    nerror=np.zeros(nf)
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros((nf,nofsnr))
    for i in range(nofsnr):
        snr=-10*np.log10(n/k)+tsnr[i]
        print(tsnr[i])
        while(np.min(nerror)<minerror):
            x_test,y_test=createTestSet(tsize,noise=True,snr=snr)
            nloop+=1
            for j in range (nf):
                nerror[j]+=calcNbrError(f[j](x_test),y_test)
        for j in range (nf):
            tber[j][i]=nerror[j]/(nloop*tsize*k)
        nerror=np.zeros(nf)
        nloop=0
    return tsnr,tber

def calcBERAE(encode,decode,minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,MAPAE=False,minsnr=MINSNR):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    tmap=np.zeros(nofsnr)
    nerrorMAP=0
    DB=[]
    if MAPAE:
        u=np.array(createAllBitvectors())
        x=encode(u)
        for i in range(2**k):
            DB.append([x[i],u[i]])
    for i in range(nofsnr):
        print(tsnr[i])
        snr=-10*np.log10(n/k)+tsnr[i]
        
        while(np.min([nerror,nerrorMAP])<minerror):
            x_test,y_test=createTestAE(encode,tsize,snr)
            nloop+=1
            nerror+=calcNbrError(decode(x_test),y_test)
            if MAPAE:
                nerrorMAP+=calcNbrError(applyMAP(x_test,DB),y_test)
            else :
                nerrorMAP=nerror
        tber[i]=nerror/(nloop*tsize*k)
        tmap[i]=nerrorMAP/(nloop*tsize*k)
        nerror=0
        nerrorMAP=0
        nloop=0
    return tsnr,tber,tmap

def calcBERUBPSK(minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR,noisetype=1):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    for i in range(nofsnr):
        snr=tsnr[i]
        print(snr)
        while(nerror<minerror):
            x_test,y_test=createTestUBPSK(tsize,snr,noisetype)
            nloop+=1
            nerror+=calcNbrError(thresoldDecoder(x_test),y_test)
        tber[i]=nerror/(nloop*tsize*k)
        nerror=0
        nloop=0
    return tsnr,tber

def calcBERHamming(minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR,noisetype=1):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    DB=[]
    u=createAllBitvectors(4)
    x=BPSK(multH(u))
    for i in range(2**4):
        DB.append([x[i],u[i]])
    for i in range(nofsnr):
        snr=-10*np.log10(7/4)+tsnr[i]
        print(tsnr[i])
        while(nerror<minerror):
            x_test,y_test=createTestHamming(tsize,snr,noisetype)
            nloop+=1
            if noisetype==1:
                nerror+=calcNbrError(applyMAP(x_test,DB,4),y_test)
            else :
                nerror+=calcNbrError(applyMAPL(x_test,DB,4),y_test)
        tber[i]=nerror/(nloop*tsize*4)
        nerror=0
        nloop=0
    return tsnr,tber

def calcBEROH(encode,decode,minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR,MAPOH=False):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    tmap=np.zeros(nofsnr)
    nerrorMAP=0
    DB=[]
    if MAPOH:
        u=np.eye(2**k)
        x=encode(u)
        for i in range(2**k):
            DB.append([x[i],OH2bin(u[i])])
    for i in range(nofsnr):
        snr=-10*np.log10(n/k)+tsnr[i]
        print(tsnr[i])
        while(np.min([nerror,nerrorMAP])<minerror):
            x_test,y_test=createTestOH(encode,tsize,snr)            
            nloop+=1
            nerror+=calcNbrError(OHtoCdws(decode(x_test)),y_test)
            if MAPOH:
                nerrorMAP+=calcNbrError(applyMAP(x_test,DB),y_test)
            else :
                nerrorMAP=nerror
        tber[i]=nerror/(nloop*tsize*k)
        tmap[i]=nerrorMAP/(nloop*tsize*k)
        nerror=0
        nerrorMAP=0
        nloop=0
    return tsnr,tber,tmap

def calcBEROHL(encode,decode,minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR,MAPOHL=False,MAPP=True):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    tmap=np.zeros(nofsnr)
    tmapp=np.zeros(nofsnr)
    nerrorMAP=0
    nerrorMAPP=0
    DB=[]
    if MAPOHL:
        u=np.eye(2**k)
        x=encode(u)
        for i in range(2**k):
            DB.append([x[i],OH2bin(u[i])])
    for i in range(nofsnr):
        snr=-10*np.log10(n/k)+tsnr[i]
        print(tsnr[i])
        while(np.min([nerror,nerrorMAP,nerrorMAPP])<minerror):
            x_test,y_test=createTestOH(encode,tsize,snr,2)   
            
            nloop+=1
            nerror+=calcNbrError(OHtoCdws(decode(x_test)),y_test)
            if MAPOHL:
                nerrorMAP+=calcNbrError(applyMAPL(x_test,DB),y_test)
            else :
                nerrorMAP=nerror
            if MAPP:
                x_testp,y_testp=createTestSet(tsize,noise=True,snr=snr,noisetype=2)
                nerrorMAPP+=calcNbrError(applyMAPL(x_testp),y_testp)
            else :
                nerrorMAPP=nerror
        tber[i]=nerror/(nloop*tsize*k)
        tmap[i]=nerrorMAP/(nloop*tsize*k)
        tmapp[i]=nerrorMAPP/(nloop*tsize*k)
        nerror=0
        nerrorMAP=0
        nerrorMAPP=0
        nloop=0
    return tsnr,tber,tmap,tmapp

def calcBERHYBRID(encode,decode,minerror=MINERROR,tsize=TESTSIZE,maxsnr=MAXSNR, nofsnr=NOFSNR,minsnr=MINSNR):
    nerror=0
    tsnr=np.linspace(minsnr,maxsnr,nofsnr)
    nloop=0
    tber=np.zeros(nofsnr)
    for i in range(nofsnr):
        snr=10*np.log(n/k)+tsnr[i]
        print(tsnr[i])
        while(nerror<minerror):
            x_test,y_test=createTestOH(encode,tsize,snr)            
            nloop+=1
            nerror+=calcNbrError(decode(x_test),y_test)
        tber[i]=nerror/(nloop*tsize*k)
        nerror=0
        nloop=0
    return tsnr,tber


def plotBER(tsnr,tber):
    plt.figure()
    if tber.ndim==1:
        plt.plot(tsnr,tber)
    
    elif len(tber)== 2:
        plt.plot(tsnr,tber[0])
        plt.plot(tsnr,tber[1],'*')
    else :
        for i in range(len(tber)):
            plt.plot(tsnr,tber[i])
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.grid()
    plt.show()

def plotWithLabels(tsnr,tber,labels):
    plt.figure()
    if tber.ndim==1:
        plt.plot(tsnr,tber)
    
    elif len(tber)== 2:
        plt.plot(tsnr,tber[0])
        plt.plot(tsnr,tber[1],'*')
    else :
        for i in range(len(tber)):
            plt.plot(tsnr,tber[i])
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.grid()
    plt.legend(labels)
    plt.show()

def save(m,names):
    for i in range(len(names)):
        np.save(names[i],m[i])
  
def load(names):
    l = []
    for n in names:
        l.append(np.load(n))
    return l

def bmatrix(a):
    text = r'$\left[\begin{array}{*{'
    text += str(len(a[0]))
    text += r'}c}'
    text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            text += str(a[x][y])
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += r'\end{array}\right]$'

    print (text)