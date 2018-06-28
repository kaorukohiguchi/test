import numpy as np
import random
import math
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from f import myf

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

def sigmoid(x):
    return 1*(1/(1+np.exp(-s*(x-x0)))-0.32)*1.35
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
    isen= 600*scipy.stats.multivariate_normal(mean,cov).pdf(pos)
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
#        for w0 in range(self.n):
#                for d0 in range(self.n):
#                    for w1 in range(self.n):
#                        for h1 in range(self.n):
#                            for d1 in range(self.n):
#                                if h1==h0 and w1==w0 and d1==d0:
#                                    self.lamda[w0,h0,d0,w1,h1,d1]=0
#                                else:
#                                    self.lamda[w0,h0,d0,w1,h1,d1]+=l_ex*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/sigma_ex**2)-l_in*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/sigma_in**2)

        self.w_cross_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
    #    for j in range(self.n):
    #        for k in range(self.n):
    #            #self.w_cross_a[k,j]=0.8*r0*np.exp(-abs(k-j)**2/(0.040*sigma_r**2))
    #            if (abs(k-j)<3):
    #                self.w_cross_a[k,j]=1.0
        self.w_cross_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
    #    for j in range(self.n):
    #        for k in range(self.n):
    #            if (abs(k-j)<3):
    #                self.w_cross_v[k,j]=1.0
            #    self.w_cross_v[k,j]=0.8*r0*np.exp(-abs(k-j)**2/(0.040*sigma_r**2))

#        for w0 in range(self.n):
#            for h0 in range(self.n):
#                for d0 in range(self.n):
#                    for w1 in range(self.n):
#                        for h1 in range(self.n):
#                            for d1 in range(self.n):
#                            self.r_field_v[w1,h1,w2,h]=r0*np.exp(-np.linalg.norm(np.array([w1,h1])-np.array([w2,h2]))**2/(2.0*sigma_r**2))   #2.41,A11
##            for h0 in range(self.n):
##                    for w1 in range(self.n):
##                            for d1 in range(self.n):
#                            self.r_field_a[w1,h1,w2,h]=r0*np.exp(-np.linalg.norm(np.array([w1,h1])-np.array([w2,h2]))**2/(2.0*sigma_r**2))
    #    print(self.w_cross_v)
        a1=np.random.randint(0, self.n)
        a2=np.random.randint(0, self.n)
        a3=np.random.randint(0, self.n)
        k=self.n
        re=np.zeros((50,self.n,self.n,self.n))
        for test in range(50):
            a1=np.random.randint(0, self.n)
            a2=np.random.randint(0, self.n)
            a3=np.random.randint(0, self.n)
            re[test,:,:,:]=inp(k,1,a1,a2,a3)
    #    plt.plot(re[:,:,3,3].T)
    #    plt.show()
    def train(self,s):
        fia=np.zeros((self.n,self.n,self.n,self.n))
        fiv=np.zeros((self.n,self.n,self.n,self.n))
        wa=np.zeros((self.n,self.n,self.n,self.n))
        wv=np.zeros((self.n,self.n,self.n,self.n))
    #    stepa=np.zeros((2,self.n,self.n))
        stepv=np.zeros((1,self.n))
        res=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n,12))
        epoch=50
        for epo in range(epoch):
            stepa=np.zeros((2,self.n,self.n,self.n))
            y=np.zeros((2,self.n,self.n,self.n))
            u=np.zeros((2,self.n,self.n,self.n))
            l=np.zeros((2,self.n,self.n,self.n))
            c=np.zeros((2,self.n,self.n,self.n))
            input=np.zeros((2,self.n,self.n,self.n))

            self.lamda=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
            for w0 in range(self.n):
                for h0 in range(self.n):
                    for d0 in range(self.n):
                        for w1 in range(self.n):
                            for h1 in range(self.n):
                                for d1 in range(self.n):
                                    if h1==h0 and w1==w0 and d1==d0:
                                        self.lamda[w0,h0,d0,w1,h1,d1]=0
                                    else:
                                        self.lamda[w0,h0,d0,w1,h1,d1]=l_ex*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/0.07*sigma_ex**2)-l_in*np.exp(-(myd(w0,h0,d0,w1,h1,d1))**2/2.0/0.1*sigma_in**2)

            self.w_cross_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
            for w0 in range(self.n):
                for h0 in range(self.n):
                    for d0 in range(self.n):
                        for w1 in range(self.n):
                            for h1 in range(self.n):
                                for d1 in range(self.n):
                                    self.w_cross_a[w0,h0,d0,w1,h1,d1]=0.45*r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(0.001*sigma_r**2))
            self.w_cross_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
    #        for w0 in range(self.n):
    #            for h0 in range(self.n):
    #                for d0 in range(self.n):
    #                    for w1 in range(self.n):
    #                        for h1 in range(self.n):
    #                            for d1 in range(self.n):
    #                                self.w_cross_v[w0,h0,d0,w1,h1,d1]=0.4*r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(0.0001*sigma_r**2))self.w_cross_a
            self.w_cross_v=np.copy(self.w_cross_a)
            self.r_field_v=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
            for w0 in range(self.n):
                for h0 in range(self.n):
                    for d0 in range(self.n):
                        for w1 in range(self.n):
                            for h1 in range(self.n):
                                for d1 in range(self.n):
                                    self.r_field_v[w0,h0,d0,w1,h1,d1]=r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(2.0*0.001*0.5*sigma_r**2))   #2.41,A181
            self.r_field_a=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
            self.r_field_a=np.copy(self.r_field_v)
    #        for w0 in range(self.n):
    #            for h0 in range(self.n):
    #                for d0 in range(self.n):
    #                    for w1 in range(self.n):
    #                        for h1 in range(self.n):
    #                            for d1 in range(self.n):
    #                                self.r_field_a[w0,h0,d0,w1,h1,d1]=r0*np.exp(-myd(w0,h0,d0,w1,h1,d1)**2/(2.0*0.001*sigma_r**2))
        #    res[:,0]=self.r_field_v[:,90]
        #    res[:,1]=self.w_cross_v[:,90]
            res[:,:,:,:,:,:,0]=self.r_field_a[:,:,:,:,:,:]
            res[:,:,:,:,:,:,1]=self.w_cross_v[:,:,:,:,:,:]
            if s==1:
                num=151
            for t in range(num):
                theta1=np.random.randint(0, self.n, (1, 3))
                a1=np.random.randint(0, self.n)
                a2=np.random.randint(0, self.n)
                a3=np.random.randint(0, self.n)
                k=self.n
                if(t<80):
                    input[0,:,:,:]=inp(k,1,a1,a2,a3)
                    input[1,:,:,:]=0
                else:#1
                    input[0,:,:,:]=inp(k,1,a1,a2,a3)
                    input[1,:,:,:]=inp(k,1,a1,a2,a3)
            #print('AAA',input)
                print('input', input)
                u[0,:,:,:]=np.sum(np.sum(np.sum(input[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.r_field_v,axis=5),axis=4),axis=3)
                u[1,:,:,:]=np.sum(np.sum(np.sum(input[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.r_field_a,axis=5),axis=4),axis=3)
                print(u)
                y=sigmoid(u)
                print('yf',y)
                l[0,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
                l[1,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
    #            print(self.lamda)
    #            print('l',l)

                c[0,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_v,axis=5),axis=4),axis=3)
                c[1,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_a,axis=5),axis=4),axis=3)
                y=sigmoid(u+c+l)
                print('u+c+l',u+c+l)
                for j in range(2):
                    y+=0.002*(-y+sigmoid(u+l+c))
                    l[0,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)
                    l[1,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.lamda,axis=5),axis=4),axis=3)

            #    print(y)
                    c[0,:,:,:]=np.sum(np.sum(np.sum(y[1,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_v,axis=5),axis=4),axis=3)
                    c[1,:,:,:]=np.sum(np.sum(np.sum(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:]*self.w_cross_a,axis=5),axis=4),axis=3)
                print('c',c)
            #    print(l)
                print('y',y)
                ytest=np.where(y>-0.2,-1,0)
                stepa=np.where(stepa>0,stepa+1,0)
                stepa=np.where(stepa+ytest==-1,1,stepa)
                print('step',stepa)
#r update  2.19,2.39,A9
                ga=myf(stepa)
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
        #        self.r_field_a=np.where(self.r_field_a<0,0,self.r_field_a)
            #    print(ga*(input_exa-self.r_field_a)*y_exa)
    #        print(self.r_field_a)
#w update 2.40,A180
        #        yj_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        #        y_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        #        yj_ex[:]=np.copy(y[0,:,:,:][np.newaxis,np.newaxis,np.newaxis,:])
        #        y_ex[:]=np.copy(y[1,:,:,:][:,np.newaxis,np.newaxis,np.newaxis])
                self.w_cross_v+=ga_ex*(y_ex-self.w_cross_v)*y_exa
                self.w_cross_a+=ga_exa*(y_exa-self.w_cross_a)*y_ex
        #        yj_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        #        y_ex=np.zeros((self.n,self.n,self.n,self.n,self.n,self.n))
        #        yj_ex[i,:]=np.copy(y[1,:,:,:])
        #        y_ex[:,i]=np.copy(y[0,:,:,:])
        #        self.w_cross_a+=ga_exa*(yj_ex-self.w_cross_a)*y_ex
                if(t==80):
                    res[:,:,:,:,:,:,2]+=self.r_field_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,3]+=self.r_field_a[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,4]+=self.w_cross_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,5]+=self.w_cross_a[:,:,:,:,:,:]/epoch
                elif(t==150):
                    res[:,:,:,:,:,:,6]+=self.r_field_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,7]+=self.r_field_a[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,8]+=self.w_cross_v[:,:,:,:,:,:]/epoch
                    res[:,:,:,:,:,:,9]+=self.w_cross_a[:,:,:,:,:,:]/epoch
        #    fia+=self.r_field_a/epoch
        #    fiv+=self.r_field_v/epoch
        #    wa+=self.w_cross_a/epoch
        #    wv+=self.w_cross_v/epoch

#        print(fia)
#        print(fiv)
    #    res=np.zeros((181,6))
    #    res[:,0]=fiv[:,30]
    #    res[:,1]=fiv[:,70]
    #    res[:,2]=fia[:,30]
    #    res[:,3]=fia[:,70]
    #    res[:,4]=wv[:,70]
    #    res[:,5]=wa[:,70]
    #    res=res/epoch
        plt.plot(res[4,4,4,:,4,4,0].T,color='black')
        plt.plot(res[4,4,4,:,4,4,1].T,color='grey')

        plt.plot(res[4,4,4,:,4,4,2].T,color='darkred')
        plt.plot(res[4,4,4,:,4,4,3].T,color='darkblue')
        plt.plot(res[4,4,4,:,4,4,4].T,color='hotpink')
        plt.plot(res[4,4,4,:,4,4,5].T,color='deepskyblue')
        plt.plot(res[4,4,4,:,4,4,6].T,color='red')
        plt.plot(res[4,4,4,:,4,4,7].T,color='blue')
        plt.plot(res[4,4,4,:,4,4,8].T,color='lightpink')
        plt.plot(res[4,4,4,:,4,4,9].T,color='skyblue')
    ##    plt.plot(res,color=['black','grey','darkred','darkblue','hotpink','black','grey','darkred','darkblue','hotpink'])
        plt.legend(['weight_0(s,v)','crossmodal_0(s,v)','weight_150(s)','weight_150(v)','crossmodal_150(s)','crossmodal_150(v)','weight_300(s)','weight_300(v)','crossmodal_300(s)','crossmodal_300(v)'])
    #    plt.title("Change of synaptic weight (hypothesis)(neulon number=90 as example)")
    #    plt.ylabel("synaptic weight")
    #    plt.xlabel("neulon number")
        plt.title("5 times-100-0.1")
        plt.show()



        plt.plot(res[4,4,4,:,4,4,0].T,color='black')
        plt.plot(res[4,4,4,:,4,4,2].T,color='darkred')
        plt.plot(res[4,4,4,:,4,4,3].T,color='darkblue')
        plt.plot(res[4,4,4,:,4,4,6].T,color='red')
        plt.plot(res[4,4,4,:,4,4,7].T,color='blue')
        plt.legend(['weight_0(s,v)','weight_150(s)','weight_150(v)','weight_300(s)','weight_300(v)'])
        plt.show()

        plt.plot(res[4,4,4,:,4,4,1].T,color='grey')
        plt.plot(res[4,4,4,:,4,4,4].T,color='hotpink')
        plt.plot(res[4,4,4,:,4,4,5].T,color='deepskyblue')
        plt.legend(['crossmodal_0(s,v)','crossmodal_0(s)','crossmodal_0(v)'])
        plt.show()



        x = y = np.arange(0, self.n, 1)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(8,10))
        ax = Axes3D(fig)
    #    Z=np.zeros((3,self.n,self.n))
        Z0=res[4,4,4,:,:,4,0]+4
        Z80=res[4,4,4,:,:,4,2]+4
        Z150=res[4,4,4,:,:,4,6]+4

        Z0_6=res[4,4,4,:,:,6,0]+6
        Z80_6=res[4,4,4,:,:,6,2]+6
        Z150_6=res[4,4,4,:,:,6,6]+6
        Z0_8=res[4,4,4,:,:,8,0]+8
        Z80_8=res[4,4,4,:,:,8,2]+8
        Z150_8=res[4,4,4,:,:,8,6]+8
        Z0_2=res[4,4,4,:,:,2,0]+2
        Z80_2=res[4,4,4,:,:,2,2]+2
        Z150_2=res[4,4,4,:,:,2,6]+2
        ax.plot_wireframe(X, Y, Z0, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_6, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_6, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_6, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_8, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_8, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_8, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_2, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_2, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_2, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.view_init(8,-60)
        plt.show()


        fig = plt.figure(figsize=(8,10))
        ax = Axes3D(fig)
        Z0=res[4,4,4,:,:,4,0]+4
        Z80=res[4,4,4,:,:,4,3]+4
        Z150=res[4,4,4,:,:,4,7]+4
        Z0_6=res[4,4,4,:,:,6,0]+6
        Z80_6=res[4,4,4,:,:,6,3]+6
        Z150_6=res[4,4,4,:,:,6,7]+6
        Z0_8=res[4,4,4,:,:,8,0]+8
        Z80_8=res[4,4,4,:,:,8,3]+8
        Z150_8=res[4,4,4,:,:,8,7]+8
        Z0_2=res[4,4,4,:,:,2,0]+2
        Z80_2=res[4,4,4,:,:,2,3]+2
        Z150_2=res[4,4,4,:,:,2,7]+2
        ax.plot_wireframe(X, Y, Z0, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_6, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_6, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_6, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_8, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_8, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_8, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_2, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z80_2, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z150_2, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.view_init(8,-60)
        plt.show()

        s=4
        fig = plt.figure(figsize=(8,10))
        ax = Axes3D(fig)

        Z0=res[s,s,s,:,:,2,1]+2
        Z1=res[s,s,s,:,:,2,4]+2
        Z2=res[s,s,s,:,:,2,8]+2
        Z0_4=res[s,s,s,:,:,4,1]+4
        Z1_4=res[s,s,s,:,:,4,4]+4
        Z2_4=res[s,s,s,:,:,4,8]+4
        Z0_6=res[s,s,s,:,:,6,1]+6
        Z1_6=res[s,s,s,:,:,6,4]+6
        Z2_6=res[s,s,s,:,:,6,8]+6
        Z0_8=res[s,s,s,:,:,8,1]+8
        Z1_8=res[s,s,s,:,:,8,4]+8
        Z2_8=res[s,s,s,:,:,8,8]+8
        ax.plot_wireframe(X, Y, Z0, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_4, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_4, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_4, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_6, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_6, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_6, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_8, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_8, rstride=1, cstride=1,color='darkred',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_8, rstride=1, cstride=1,color='red',linewidth = 0.7)
        ax.view_init(8,-60)
        plt.show()

        fig = plt.figure(figsize=(8,10))
        ax = Axes3D(fig)

        Z0=res[s,s,s,:,:,2,1]+2
        Z1=res[s,s,s,:,:,2,5]+2
        Z2=res[s,s,s,:,:,2,9]+2
        Z0_4=res[s,s,s,:,:,4,1]+4
        Z1_4=res[s,s,s,:,:,4,5]+4
        Z2_4=res[s,s,s,:,:,4,9]+4
        Z0_6=res[s,s,s,:,:,6,1]+6
        Z1_6=res[s,s,s,:,:,6,5]+6
        Z2_6=res[s,s,s,:,:,6,9]+6
        Z0_8=res[s,s,s,:,:,8,1]+8
        Z1_8=res[s,s,s,:,:,8,5]+8
        Z2_8=res[s,s,s,:,:,8,9]+8
        ax.plot_wireframe(X, Y, Z0, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_4, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_4, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_4, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_6, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_6, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_6, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z0_8, rstride=1, cstride=1,color='black',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z1_8, rstride=1, cstride=1,color='darkblue',linewidth = 0.7)
        ax.plot_wireframe(X, Y, Z2_8, rstride=1, cstride=1,color='blue',linewidth = 0.7)
        ax.view_init(8,-60)
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        Z0=res[4,4,4,:,:,4,1]
        Z80=res[4,4,4,:,:,4,4]
        Z150=res[4,4,4,:,:,4,8]
        Z80_7=res[4,4,4,:,:,7,4]+3
        Z150_7=res[4,4,4,:,:,7,8]+3
        ax.plot_wireframe(X, Y, Z0, rstride=1, cstride=1,color='black',linewidth = 0.8)
        ax.plot_wireframe(X, Y, Z80, rstride=1, cstride=1,color='darkblue',linewidth = 0.8)
        ax.plot_wireframe(X, Y, Z150, rstride=1, cstride=1,color='blue',linewidth = 0.8)
        ax.plot_wireframe(X, Y, Z80_7, rstride=1, cstride=1,color='darkblue',linewidth = 0.8)
        ax.plot_wireframe(X, Y, Z150_7, rstride=1, cstride=1,color='blue',linewidth = 0.8)
        return 1


if __name__ =="__main__":
    print('a')
    layer=Myclass()
    result=layer.train(1)

    trange=np.arange(0,181,1)
