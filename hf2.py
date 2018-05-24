import numpy as np
import random
import matplotlib.pyplot as plt

def mysign(x):
    return np.where(x<0,-1,1)

data1 = [[-1,-1,1,-1,-1,
        -1,1,-1,1,-1,
        -1,1,-1,1,-1,
        1,1,1,1,1,
        1,-1,-1,-1,1],
        [1,1,1,-1,-1,
        1,-1,1,-1,-1,
        1,1,1,1,-1,
        1,-1,-1,1,-1,
        1,1,1,1,-1,],
        [1,1,1,-1,-1,
        1,-1,-1,-1,-1,
        1,-1,-1,-1,-1,
        1,-1,-1,-1,-1,
        1,1,1,-1,-1],
        [1,1,1,-1,-1,
        1,-1,-1,1,-1,
        1,-1,-1,1,-1,
        1,-1,-1,1,-1,
        1,1,1,-1,-1],
        [1,1,1,1,-1,
        1,-1,-1,-1,-1,
        1,1,1,1,-1,
        1,-1,-1,-1,-1,
        1,1,1,1,-1],
        [1,1,1,1,-1,
        1,-1,-1,-1,-1,
        1,1,1,-1,-1,
        1,-1,-1,-1,-1,
        1,-1,-1,-1,-1]
        ]


class hpf:
    def __init__(self,data,error,n):
        self.data=np.array(data)
        self.data=self.data[:n,:]
        self.n=self.data.shape[0]
        self.l=self.data.shape[1]
        self.c=np.zeros((self.l,self.l))
        for i in range(self.n):
            self.c+=(np.matrix(self.data[i,:]).T*np.matrix(self.data[i,:]))

        self.w=(self.c-np.diag(np.diag(self.c)))/float(self.n)

        self.error=error

    def update(self):
        self.th=0.0

        rate=0.0
        for b in range(10):
            test = np.copy(self.data)
            a=random.randint(0,self.n-1)
            if (self.error>0):
                for i in range(self.l-1):
                    s=random.randint(0,99)
                    if(s<self.error):
                            test[a,i]=-test[a,i]
            print(test.shape)
            test=test[a,:]
            while a in range(10):
                f=mysign(np.dot(np.matrix(test),self.w)-self.th)
                if np.all(f==test):
                    if np.all(f==self.data[a]):
                        print("correct")
                        rate+= 1.0/10
                        break
                    else:
                        for j in range(self.l-1):
                            if(f[0,j]==self.data[a,j]):
                                rate+=1.0/self.l/10
                        break
                else:
                    test=f
        return rate

    def energy(self,test,w):
        return -np.sum(w*np.dot(self.data.T,self.data))/2+np.sum(test*self.th)

if __name__ =="__main__":
#    hf=hpf(data1,0,2)
#    hf.update()
    res=np.zeros((21,6))
    res2=np.zeros((21,6))
    for k in range (6):
        for i in range(21):
            hf=hpf(data1,i,k+1)
            print(i,k)
            for j in range(2):
                res[i,k]+=hf.update()/2.0



    trange=np.arange(0,21,1)
    plt.plot(trange,res)
    plt.show()
