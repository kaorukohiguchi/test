from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np

#fig, ax = plt.subplots(1, 1)

dfn, dfd = 5, 5
mean, var, skew, kurt = f.stats(dfn, dfd, moments='mvsk')
a=np.zeros((400,2))
b=np.zeros((400,1))
for i in range (360):
    a[i+40,0]=0.05*f.pdf(i*0.024, dfn, dfd, loc=0, scale=1)
for i in range (400):

    a[i,1]=0.014
#print(a)


#plt.plot(a[:,0],label='hypothesis')
#plt.plot(a[:,1],label='previous')
#plt.legend()
#plt.title("Learning rate of synaptic weight")
#plt.ylabel("learning rate")
#plt.xlabel("step")
#plt.show()

def myf(x):
    return 0.05*f.pdf(0.0001+x*0.024, dfn, dfd, loc=0, scale=1)
