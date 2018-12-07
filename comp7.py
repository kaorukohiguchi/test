import numpy as np
import random
import math
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from f import myf


font=27
plt.rcParams['font.size'] = 23 #フォントの大きさ
#parameters value
ias=36
ivs0=20
sigma_a=20
sigma_v=4
ep=7
a=0.78
eta_train=0.5
eta_test=0.33
lv=30
lav=1
x0=5.7
s=0.28
tau=5.0/1000
l_in=1.85/5.0
l_ex=1.9/2.0
sigma_ex=12
sigma_in=24
r0=1.3/1.1
sigma_r=30
ga=0.03
b=180**(-16)
xx=0.2576011
xt=3
alp=0.8
epoch=10
cularray=np.zeros((2,epoch+1))
def sigmoid(x):
    return 1*(1/(1+np.exp(-s*(x-x0)))-xx)*(1/1-xx)

#    return np.where(x<0,-1.0,1.0)
def cdistance(s,t):   #2
    if(abs(s-t)>90):
        return 180-abs(s-t)
    else:
        return abs(s-t)

def myd(x,y,z,x0,y0,z0):    #2point distance
    return  np.linalg.norm(np.array([x0,y0,z0])-np.array([x,y,z]))
def Decay(s):  #3
    return 2.1+0.058*s+0.022*s**2
    #-0.00022*s**3

def Acuity(s):  #4
    return 60/(np.sqrt(3)*Decay(s))

def vsigma(s):  #5
    return sigma_v+ep*np.sqrt(3)*(Decay(s)-Decay(0))/60

def ivs(s):  #6
    return vsigma(s)*ivs0/(sigma_v+a*(vsigma(s)-sigma_v))

#prior
def p_v(s):
    return np.exp(-(s-90)**2/(2*lv**2))/(lv*np.sqrt(2*math.pi))
def p_a(s):
    return 1/180.0
def p_aandv(a,v):
    return 0.5*p_v(v)*p_av(a,v)+0.5*p_a(a)*p_av(a,v)
#conditional probability
def p_av(a,v):
    return b/180+(1-b)*np.exp(-cdistance(a,v)**2/2/lav**2)/np.sqrt(2*math.pi*lav**2)

#1
def inp(k,s,d,w,l):
    i=0
    if (s==1):
        i=ias
        sigma=sigma_a
    else :
        i=ivs(theta-90)
        sigma=vsigma(t)
#    n=np.random.normal(0,0.1*eta_train*i/np.sqrt(2*math.pi*sigma**2))
    n=0
    x = np.linspace(0.0,k-1,k)
    y = np.linspace(0.0,k-1,k)
    z = np.linspace(0.0,k-1,k)
#    print(x)
    X,Y,Z=np.meshgrid(x,y,z)
    pos = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T
#    print(pos)
    isensor=np.zeros((k,k,k))
    mean = np.array([d,w,l])
    cov  = 0.3*np.diag([1.0,1.0,1.0])
    isen= 570*scipy.stats.multivariate_normal(mean,cov).pdf(pos)
    isen=np.reshape(isen,(k,k,k))
#    print('i', isen)
#    print(d,w,l)
    return isen

def p_is(t,s):   #8
    v=0.5
    if s==1:
        i=ias
        sigma=sigma_a
    else :
        i=ivs(t-90)
        sigma=vsigma(t)
    return 1
    #1/np.sqrt(2*math.pi*v**2)*np.exp(-(input(t,s)-istrength/np.sqrt(2*math.pi*sigma**2)))

# stimuli position
def vpos():
    n=round(random.gauss(90,math.sqrt(30)))
    if n<0:
        return 0
    elif n>180:
        return 180
    else:
        return n

def apos():
    return random.randrange(0,180)

def avpos():
    s=np.random.randint(0,2)
    return s*vpos()+(1-s)*apos()

class Myclass:
    def __init__(self):
        self.n=10

        self.r_field_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        self.r_field_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        self.lamda=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))  #v,a共通
        print('b')

        self.w_cross_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        self.w_cross_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        a1=np.random.randint(0, self.n)
        a2=np.random.randint(0, self.n)
        a3=np.random.randint(0, self.n)
        k=self.n
        re=np.zeros((50,self.n,self.n,self.n))

    #    inself.r_field_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        self.inr_field_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        self.inlamda=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))  #v,a共通
        print('b')

        self.inw_cross_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
    #    inself.w_cross_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        for w0 in range(self.n):
            for h0 in range(self.n):
                for d0 in range(self.n):
                    for w1 in range(self.n):
                        for h1 in range(self.n):
                            for d1 in range(self.n):
                                if h1==h0 and w1==w0 and d1==d0:
                                    self.inlamda[w0,h0,d0,w1,h1,d1]=0
                                else:
                                    self.inlamda[w0,h0,d0,w1,h1,d1]=l_ex*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/0.07*sigma_ex**2)-l_in*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/0.1*sigma_in**2)
        for w0 in range(self.n):
            for h0 in range(self.n):
                for d0 in range(self.n):
                    for w1 in range(self.n):
                        for h1 in range(self.n):
                            for d1 in range(self.n):
                                self.inw_cross_a[w0,h0,d0,w1,h1,d1]=0.45*r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(0.001*sigma_r**2))
        for w0 in range(self.n):
            for h0 in range(self.n):
                for d0 in range(self.n):
                    for w1 in range(self.n):
                        for h1 in range(self.n):
                            for d1 in range(self.n):
                                self.inr_field_a[w0,h0,d0,w1,h1,d1]=r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(2.0*0.001*0.5*sigma_r**2))   #2.41,A181

    #    for test in range(50):
    #        a1=np.random.randint(0, self.n)
    #        a2=np.random.randint(0, self.n)
    #        a3=np.random.randint(0, self.n)
    #        re[test,:,:,:]=inp(k,1,a1,a2,a3)
    #    plt.plot(re[:,3,3,:].T)
    #    plt.show()
    def train(self,s):
        fia=np.zeros((self.n,self.n,self.n,self.n))
        fiv=np.zeros((self.n,self.n,self.n,self.n))
        wa=np.zeros((self.n,self.n,self.n,self.n))
        wv=np.zeros((self.n,self.n,self.n,self.n))
    #    stepa=np.zeros((2,self.n,self.n))
        stepv=np.zeros((1,self.n))
        res=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n,5,17))

        for epo in range(epoch):
            print(epo)
            stepa=np.zeros((2,self.n,self.n,self.n))
            y=np.zeros((2,self.n,self.n,self.n))
            u=np.zeros((2,self.n,self.n,self.n))
            l=np.zeros((2,self.n,self.n,self.n))
            c=np.zeros((2,self.n,self.n,self.n))
            input=np.zeros((2,self.n,self.n,self.n))

            self.lamda=np.copy((self.inlamda))

            self.w_cross_v=np.copy(self.inw_cross_a)
            self.w_cross_a=np.copy(self.inw_cross_a)

            self.r_field_a=np.copy(self.inr_field_a)
            self.r_field_v=np.copy(self.inr_field_a)
        #    res[:,0]=self.r_field_v[:,90]
        #    res[:,1]=self.w_cross_v[:,90]
            res[:,:,:,:,:,:,0,0]=self.r_field_a[:,:,:,:,:,:]
            res[:,:,:,:,:,:,1,0]=self.r_field_a[:,:,:,:,:,:]
            res[:,:,:,:,:,:,2,0]=self.w_cross_v[:,:,:,:,:,:]
            res[:,:,:,:,:,:,3,0]=self.w_cross_v[:,:,:,:,:,:]
            res[:,:,:,:,:,:,4,0]=self.lamda[:,:,:,:,:,:]
            if s==1:
                num=161
            for t in range(num):
                theta1=np.random.randint(0, self.n, (1, 3))
                a1=np.random.randint(0, self.n)
                a2=np.random.randint(0, self.n)
                a3=np.random.randint(0, self.n)
                k=self.n

                input[0,:,:,:]=inp(k,1,a1,a2,a3)
                input[1,:,:,:]=inp(k,1,a1,a2,a3)
            #print('AAA',input)
            #    print('input', input)
                u[0,:,:,:]=np.sum(np.sum(np.sum(input[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.r_field_v,axis=5),axis=4),axis=3)
                u[1,:,:,:]=np.sum(np.sum(np.sum(input[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.r_field_a,axis=5),axis=4),axis=3)
                print(u)
                y=sigmoid(u)
            #    print('yf',y)
                l[0,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
                l[1,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
    #            print(self.lamda)
    #            print('l',l)

                c[0,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_v,axis=5),axis=4),axis=3)
                c[1,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_a,axis=5),axis=4),axis=3)
                y=sigmoid(u+c+l)
                print('u+c+l',u+c+l)
                for j in range(xt):
                    y+=0.002*(-y+sigmoid(u+l+c))
                    l[0,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
                    l[1,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)

            #    print(y)
                    c[0,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_v,axis=5),axis=4),axis=3)
                    c[1,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_a,axis=5),axis=4),axis=3)
            #    print('c',c)
            #    print(l)
                print('y',y)
                ytest=np.where(y>-0.14,-1,0)
                stepa=np.where(stepa>0,stepa+1,0)
                stepa=np.where(stepa+ytest==-1,1,stepa)
                print('step',stepa)
#r update  2.19,2.39,A9
                ga=0.8*myf(stepa)
                input_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                y_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                ga_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                input_ex[:]=np.copy((input[0,:,:,:])[np.newaxis,np.newaxis,np.newaxis,:])
                y_ex[:]=np.copy(y[0,:,:,:][:,np.newaxis,np.newaxis,np.newaxis])
                ga_ex[:]=np.copy(ga[0,:,:,:][:,np.newaxis,np.newaxis,np.newaxis])
    #        print(ga*(i  nput_ex-self.r_field_v)*y_ex)
                self.r_field_v+=ga_ex*(input_ex-self.r_field_v)*y_ex

                ga_exa=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                input_exa=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                y_exa=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
                input_exa[:]=np.copy((input[1,:,:,:])[np.newaxis,np.newaxis,np.newaxis,:])
                y_exa[:]=np.copy(y[1,:,:,:][:,np.newaxis,np.newaxis,np.newaxis])
                ga_exa[:]=np.copy(ga[1,:,:,:][:,np.newaxis,np.newaxis,np.newaxis])
        #        print(y_exa)
                self.r_field_a+=ga_exa*(input_ex-self.r_field_v)*y_exa

#w update 2.40,A180
                self.w_cross_v+=ga_ex*(y_ex-self.w_cross_v)*y_exa
                self.w_cross_a+=ga_exa*(y_exa-self.w_cross_a)*y_ex
                if(t%10==0):
                    ele=0
                    ele=int(t/10)
                    res[:,:,:,:,:,:,0,ele]+=self.r_field_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,1,ele]+=self.r_field_a[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,2,ele]+=self.w_cross_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,3,ele]+=self.w_cross_a[:,:,:,:,:,:]/epoch

                if(t==80):
                    for ka in range(10):
                        for kb in range(10):
                            for kc in range(10):
                                cularray[0,epoch]+=self.r_field_v[ka,kb,kc,ka,kb,kc]/1000
                                cularray[1,epoch]+=self.r_field_a[ka,kb,kc,ka,kb,kc]/1000


    #    plt.plot(res[4,4,4,:,4,4,0,0].T,color='dimgrey')
    #    plt.plot(res[4,4,4,:,4,4,1,0].T,color='grey')

    #    plt.plot(res[4,4,4,:,4,4,0,5].T,color='darkred')
    #    plt.plot(res[4,4,4,:,4,4,1,4].T,color='darkblue')
    #    plt.plot(res[4,4,4,:,4,4,2,4].T,color='hotpink')
    #    plt.plot(res[4,4,4,:,4,4,3,5].T,color='deepskyblue')
    #    plt.plot(res[4,4,4,:,4,4,0,9].T,color='red')
    #    plt.plot(res[4,4,4,:,4,4,1,9].T,color='blue')
    #    plt.plot(res[4,4,4,:,4,4,2,9].T,color='lightpink')
    #    plt.plot(res[4,4,4,:,4,4,3,9].T,color='skyblue')
    ##    plt.plot(res,color=['dimgrey','grey','darkred','darkblue','hotpink','dimgrey','grey','darkred','darkblue','hotpink'])
    #    plt.legend(['weight_0(s,v)','crossmodal_0(s,v)','weight_150(s)','weight_150(v)','crossmodal_150(s)','crossmodal_150(v)','weight_300(s)','weight_300(v)','crossmodal_300(s)','crossmodal_300(v)'])
    #    plt.title("Change of synaptic weight (hypothesis)(neulon number=90 as example)")
    #    plt.ylabel("synaptic weight")
    #    plt.xlabel("neulon number")
    #    plt.title("5 times-100-0.29")

    #    plt.show()

        resfn=np.zeros((5,19))
        xtrange=np.arange(-5,6,1)
        for rf in range (10):
            resfn[0,9-rf:19-rf]+=res[rf,rf,rf,:,rf,rf,0,8]/10
            resfn[1,9-rf:19-rf]+=res[rf,rf,rf,:,rf,rf,1,8]/10
            resfn[2,9-rf:19-rf]+=res[rf,rf,rf,:,rf,rf,0,16]/10
            resfn[3,9-rf:19-rf]+=res[rf,rf,rf,:,rf,rf,1,16]/10
            resfn[4,9-rf:19-rf]+=res[rf,rf,rf,:,rf,rf,0,0]/10
    #        resfn[0,9-rf:19-rf]+=res[rf,rf,rf,rf,:,rf,0,8]/20
    #        resfn[1,9-rf:19-rf]+=res[rf,rf,rf,rf,:,rf,1,8]/20
    #        resfn[2,9-rf:19-rf]+=res[rf,rf,rf,rf,:,rf,0,16]/20
    #        resfn[3,9-rf:19-rf]+=res[rf,rf,rf,rf,:,rf,1,16]/20
    #        resfn[4,9-rf:19-rf]+=res[rf,rf,rf,rf,:,rf,0,0]/20
    #        resfn[0,9-rf:19-rf]+=res[rf,rf,rf,rf,rf,:,0,8]/30
    #        resfn[1,9-rf:19-rf]+=res[rf,rf,rf,rf,rf,:,1,8]/30
    #        resfn[2,9-rf:19-rf]+=res[rf,rf,rf,rf,rf,:,0,16]/30
    #        resfn[3,9-rf:19-rf]+=res[rf,rf,rf,rf,rf,:,1,16]/30
    #        resfn[4,9-rf:19-rf]+=res[rf,rf,rf,rf,rf,:,0,0]/30
        resf=resfn[:,4:15]
    #    plt.plot(res[4,4,4,:,4,4,0].T,color='dimgrey')
    #    plt.legend(fontsize=18)
        plt.plot(xtrange,resf[0],color='darkred')
        plt.plot(xtrange,resf[1],color='darkblue')
        plt.plot(xtrange,resf[2],color='deeppink')
        plt.plot(xtrange,resf[3],color='deepskyblue')
        plt.plot(xtrange,resf[4],color='dimgrey')
    #    plt.show()
    #    plt.plot(xtrange,resf,color='darkblue')
    #    plt.plot(res[4,4,4,:,4,4,6].T,color='red')
    #    plt.plot(res[4,4,4,:,4,4,7].T,color='blue')
        plt.ylabel("synaptic weight",fontsize=15)
        plt.ylim(-0.2, 1.8)
        plt.xlabel("position",fontsize=15)
        plt.tick_params(labelsize=14)
    #    plt.legend(fontsize=22)
        plt.legend(['weight_80(s)','weight_80(v)','weight_160(s)','weight_160(v)','init'],fontsize=14)
        plt.savefig('figure-r_m'+str(xx)+str(xt)+'.png')
        plt.close()

        plt.plot(res[4,4,4,:,4,4,0,0].T,color='dimgrey')
        plt.plot(res[4,4,4,:,4,4,0,8].T,color='darkred')
        plt.plot(res[4,4,4,:,4,4,1,8].T,color='darkblue')
        plt.plot(res[4,4,4,:,4,4,0,16].T,color='red')
        plt.plot(res[4,4,4,:,4,4,1,16].T,color='deepskyblue')
        plt.legend(['weight_0(s,v)','weight_80(s)','weight_80(v)','weight_160(s)','weight_160(v)'])
        plt.savefig('figure-r'+str(xx)+str(xt)+'.jpg')
    #    plt.show()


        plt.plot(res[4,4,4,:,4,4,2,0].T,color='grey')
        plt.plot(res[4,4,4,:,4,4,2,8].T,color='darkred')
        plt.plot(res[4,4,4,:,4,4,3,8].T,color='darkblue')
        plt.plot(res[4,4,4,:,4,4,2,16].T,color='red')
        plt.plot(res[4,4,4,:,4,4,3,16].T,color='deepskyblue')
        plt.legend(['crossmodal_0(s,v)','crossmodal_80(s)','crossmodal_80(v)'])
        plt.savefig('figure-cross'+str(xx)+str(xt)+'.jpg')
    #    plt.show()

        for i in range(17):
            x = y = np.arange(0, self.n, 1)
            X, Y = np.meshgrid(x, y)
            fig = plt.figure(figsize=(8,8))
            ax = Axes3D(fig)
    #    Z=np.zeros((3,self.n,self.n))
            Z0=(res[4,4,4,4,:,:,0,i]-np.min(res[4,4,4,4,:,:,0,i])*0.6)*0.65+4
            Z0_5=(res[4,4,4,5,:,:,0,i]-np.min(res[4,4,4,5,:,:,0,i])*0.6)*0.65+5
            Z0_3=(res[4,4,4,3,:,:,0,i]-np.min(res[4,4,4,3,:,:,0,i])*0.6)*0.65+3
            ax.plot_surface(X,Y,Z0,alpha=alp,cmap=cm.Reds,vmin=3.9,vmax=4.7)
            ax.plot_surface(X,Y,Z0_5,alpha=alp,cmap=cm.Reds,vmin=4.9,vmax=5.7)
            ax.plot_surface(X,Y,Z0_3,alpha=alp,cmap=cm.Reds,vmin=2.9,vmax=3.7)
            plt.title(str(i*10)+"step",fontsize=font)
            ax.set_zlim(2.7,5.6)
            plt.legend()
            ax.view_init(6,-60)
            #plt.colorbar(surf,shrink=.17,aspect=4)
    #        plt.clim(-0.5,2)
            plt.savefig("sfigs/"+str(i)+".jpg")
            plt.close()
        #plt.show()
            x = y = np.arange(0, self.n, 1)
            X, Y = np.meshgrid(x, y)

            fig = plt.figure(figsize=(8,8))
            ax = Axes3D(fig)
            Z0=(res[4,4,4,4,:,:,1,i]-res[4,4,4,4,8,8,1,i]*0.5)*0.65+4
            Z0_5=(res[4,4,4,5,:,:,1,i]-res[4,4,4,5,8,8,1,i]*0.5)*0.65+5
            Z0_3=(res[4,4,4,3,:,:,1,i]-res[4,4,4,3,8,8,1,i]*0.5)*0.65+3
            ax.plot_surface(X,Y,Z0,alpha=alp,cmap=cm.Reds,vmin=3.9,vmax=4.7)
            ax.plot_surface(X,Y,Z0_5,alpha=alp,cmap=cm.Reds,vmin=4.9,vmax=5.7)
            ax.plot_surface(X,Y,Z0_3,alpha=alp,cmap=cm.Reds,vmin=2.9,vmax=3.7)
            plt.legend()
            ax.set_zlim(2.7,5.6)
            ax.view_init(6,-60)
            plt.title(str(i*10)+"step",fontsize=font)
            plt.savefig("vfigs/"+str(i)+".jpg")
            plt.close()

            fig = plt.figure(figsize=(8,6))
            ax = Axes3D(fig)
            Z0=res[4,4,4,4,:,:,2,i]+4-res[4,4,4,4,0,0,2,i]
            Z0_5=res[4,4,4,5,:,:,2,i]+5-res[4,4,4,5,0,0,2,i]
            Z0_3=res[4,4,4,3,:,:,2,i]+3-res[4,4,4,3,0,0,2,i]
            ax.plot_surface(X,Y,Z0,alpha=alp,cmap=cm.Reds,vmin=3.95,vmax=4.4)
            ax.plot_surface(X,Y,Z0_5,alpha=alp,cmap=cm.Reds,vmin=4.95,vmax=5.4)
            ax.plot_surface(X,Y,Z0_3,alpha=alp,cmap=cm.Reds,vmin=2.95,vmax=3.4)
        #    surfc=ax.plot_surface(X,Y,Z0_0,alpha=alp,cmap=cm.Reds,vmin=0,vmax=0.5)
            plt.legend()
            ax.view_init(6,-60)
            ax.set_zlim(2.8,5.4)
        #    plt.colorbar(surfc,shrink=.18,aspect=3)
        #    plt.clim(-0.5,1.5)
            plt.title(str(i*10)+"step",fontsize=font)
            plt.savefig("csfigs/"+str(i)+".jpg")
            plt.close()

            fig = plt.figure(figsize=(8,6))
            ax = Axes3D(fig)
            Z0c=res[4,4,4,4,:,:,3,i]+4-res[4,4,4,4,0,0,3,i]
            Z0_5c=res[4,4,4,5,:,:,3,i]+5-res[4,4,4,5,0,0,3,i]
            Z0_3c=res[4,4,4,3,:,:,3,i]+3-res[4,4,4,3,0,0,3,i]
            ax.plot_surface(X,Y,Z0c,alpha=alp,cmap=cm.Reds,vmin=3.95,vmax=4.4)
            ax.plot_surface(X,Y,Z0_5c,alpha=alp,cmap=cm.Reds,vmin=4.95,vmax=5.4)
            ax.plot_surface(X,Y,Z0_3c,alpha=alp,cmap=cm.Reds,vmin=2.95,vmax=3.4)
            plt.legend()
            ax.set_zlim(2.8,5.4)
            ax.view_init(6,-60)
            plt.title(str(i*10)+"step",fontsize=font)
            plt.savefig("cvfigs/"+str(i)+".jpg")
            plt.close()

            fig = plt.figure(figsize=(8,6))
            ax = Axes3D(fig)
            Z0=res[4,4,4,4,:,:,2,i]+4
            Z0_5=res[4,4,4,5,:,:,2,i]+5
            Z0_3=res[4,4,4,3,:,:,2,i]+3
            ax.plot_surface(X,Y,Z0,alpha=alp,cmap=cm.Reds,vmin=3.95,vmax=4.5)
            ax.plot_surface(X,Y,Z0_5,alpha=alp,cmap=cm.Reds,vmin=4.95,vmax=5.5)
            ax.plot_surface(X,Y,Z0_3,alpha=alp,cmap=cm.Reds,vmin=2.95,vmax=3.5)
        #    surfc=ax.plot_surface(X,Y,Z0_0,alpha=alp,cmap=cm.Reds,vmin=0,vmax=0.5)
            plt.legend()
            ax.view_init(6,-60)
            ax.set_zlim(2.8,5.7)
        #    plt.colorbar(surfc,shrink=.18,aspect=3)
        #    plt.clim(-0.5,1.5)
            plt.title(str(i*10)+"step",fontsize=font)
            plt.savefig("abscsfigs/"+str(i)+".jpg")
            plt.close()

            fig = plt.figure(figsize=(8,6))
            ax = Axes3D(fig)
            Z0=res[4,4,4,4,:,:,3,i]+4
            Z0_5=res[4,4,4,5,:,:,3,i]+5
            Z0_3=res[4,4,4,3,:,:,3,i]+3
            ax.plot_surface(X,Y,Z0,alpha=alp,cmap=cm.Reds,vmin=3.95,vmax=4.5)
            ax.plot_surface(X,Y,Z0_5,alpha=alp,cmap=cm.Reds,vmin=4.95,vmax=5.5)
            ax.plot_surface(X,Y,Z0_3,alpha=alp,cmap=cm.Reds,vmin=2.95,vmax=3.5)
        #    surfc=ax.plot_surface(X,Y,Z0_0,alpha=alp,cmap=cm.Reds,vmin=0,vmax=0.5)
            plt.legend()
            ax.view_init(6,-60)
            ax.set_zlim(2.8,5.7)
        #    plt.colorbar(surfc,shrink=.18,aspect=3)
        #    plt.clim(-0.5,1.5)
            plt.title(str(i*10)+"step",fontsize=font)
            plt.savefig("abscvfigs/"+str(i)+".jpg")
            plt.close()
        plt.close()

    #    plt.show()
        return 1


if __name__ =="__main__":
    print('a')
    layer=Myclass()
    result=layer.train(1)
    print(np.average(cularray,axis=1))
    print(np.var(cularray,axis=1))
    trange=np.arange(0,181,1)
