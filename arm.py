import numpy as np
import math

def armpos():
    arm=np.zeros((1,3))
    sh1=random.randrange(-math.pi/2,math.pi/2)
    sh2=random.randrange(-math.pi/2,math.pi/2)
    sh3=random.randrange(-math.pi/2,math.pi/2)
    el1=random.randrange(-math.pi/2,math.pi/2)
    el2=random.randrange(-math.pi/2,math.pi/2)

    arm=np.array([6*math.cos(sh1),6*math.sin(sh1)*6*math.sin(sh2),6*math.sin(sh1)*6*math.sin(sh2)])   #z,y,x
