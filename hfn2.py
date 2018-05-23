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
    #    print(self.data.shape[1])
    #    print(np.dot(self.data.T,self.data)/self.n)
        self.c=np.zeros((self.l,self.l))
        for i in range(self.n):
        #    print(self.data[i].T)
        #    print(self.data[i])
            self.c+=(np.matrix(self.data[i,:]).T*np.matrix(self.data[i,:]))
        #    print(self.c)
        self.w=(self.c-np.diag(np.diag(self.c)))/float(self.n)
    #    print(self.w)
        self.error=error

    def update(self):
        self.th=0.0
        # debug param
        data_check = np.copy(self.data)
        test = np.copy(self.data)
        a=random.randint(0,self.n-1)
        if (self.error>0):
            for i in range(self.l-1):
                s=random.randint(0,99)
                if(s<self.error):
                        test[a,i]=-test[a,i]
        test=test[a,:]
        print("IS SAME?", (data_check[a] == self.data[a]).all())
        import ipdb; ipdb.set_trace()
        while True:
            f=mysign(np.dot(np.matrix(test),self.w)-self.th)
            if np.all(f==test):
                if np.all(f==self.data[a]):
                    print("correct")
                    return 1.0
                    break
                else:
                    rate=0.0;
                    print("DATA",a,self.data[a])
                    for j in range(self.l-1):
                        if(f[0,j]==self.data[a,j]):
                            rate+=1.0/self.l
                    return float(rate)
                    break
            else:
                test=f
            #    print(test)

        e=self.energy(test,self.w)
        rate=0.0;
        for j in range(self.l-1):
            if(f[0,j]==self.data[a,j]):
                rate+=1.0/self.l
        return float(rate)
    #    print(f.shape)

    def energy(self,test,w):
        return -np.sum(w*np.dot(self.data.T,self.data))/2+np.sum(test*self.th)

if __name__ =="__main__":
    hf=hpf(data1,0,2)
    hf.update()
    res=np.zeros((51,1))
    for i in range (5):
        r=0.0
        for j in range(50):
            hf=hpf(data1,i,4)
            r+=float(hf.update())
        res[i,0]=r/50.0

    trange=np.arange(0,51,1)
    plt.plot(trange,res)
    plt.show()
