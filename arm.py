import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

arm=np.zeros((1,3))
def armcul():
    arm=np.zeros((1,3))
    sh3=random.uniform(0,math.pi)
    sh2=random.uniform(-math.pi/2,math.pi/2)
    sh1=random.uniform(-math.pi/2,math.pi/2)
    el=random.uniform(-math.pi/2,math.pi/2)
    l=5
    c1=math.cos(sh1)
    s1=math.sin(sh1)
    c2=math.cos(sh2)
    s2=math.sin(sh2)
    c3=math.cos(sh3)
    s3=math.sin(sh3)
    ce=math.cos(el)
    se=math.sin(el)
    #arm=np.array([6*math.cos(sh1),6*math.sin(sh1)*6*math.sin(sh2),6*math.sin(sh1)*6*math.sin(sh2)])  #z,y,x
    mat1=np.zeros((4,4))
    mat2=np.zeros((4,4))

    #mat1=np.array([[c1*c2*c3-s1*s3,-c1*c2*s3-s1*c3,c1*s2,l],[s1*c2*c3-c1*s3,-s1*c2*s3+c1*c3,s1*s2,0],[-s2*c3,s2*s3,c2,0],[0,0,0,1]])
    mat1=np.array([[c2*c3,s1*s2*c3+c1*s3,-c1*s2*c3+s1*s3,l],[-c2*s3,-s1*s2*s3+c1*c3,c1*s2*s3+s1*c3,0],[s2,-s1*c2,c1*c2,0],[0,0,0,1]])
    mat2=np.array([[ce,-se,0,0],[se,ce,0,0],[0,0,1,0],[0,0,0,1]])
    #print()
    matr=np.array([[c2*c3,s1*s2*c3+c1*s3,-c1*s2*c3+s1*s3,0],[-c2*s3,-s1*s2*s3+c1*c3,c1*s2*s3+s1*c3,l],[s2,-s1*c2,c1*c2,0],[0,0,0,1]])
    armmat=np.dot(mat2,mat1)
    init=np.array([[l],[0],[0],[1]])
    m1=np.dot(matr,init)
    print(m1)
    m2=np.dot(mat2,m1)
    print(m2)
    arm=m2[:3]
    return np.round(np.array(arm))

def armpos():
    while True:
        a=armcul()
        if(a[0]>-5 and a[0]<6 and a[1]>-5 and a[1]<6 and a[2]>-5 and a[2]<6):
            break
    return a+4


if __name__ =="__main__":
    x=np.zeros((3,700))

    for i in range(700):
        x[0,i]=armpos()[0]
        x[1,i]=armpos()[1]
        x[2,i]=armpos()[2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[0,:],x[1,:],x[2,:])
    ax.set_xlabel('x')  # X軸ラベル
    ax.set_ylabel('y')  # Y軸ラベル
    ax.set_zlabel('z')
    plt.show()
