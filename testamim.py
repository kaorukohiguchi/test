import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm


fig = plt.figure()
ax = Axes3D(fig)

x = y = np.arange(0, 10, 1)
X, Y = np.meshgrid(x, y)

Z=np.zeros((2,10,10,20))

for s in range(10):
    for t in range(10):
        for i in range(20):
            Z[0,s,t,i]=s**2+t**2+3*i
            Z[1,s,t,i]=s**2-t

#ax.plot_surface(X, Y, Z[0,X,Y,2], rstride=1, cstride=1, cmap=cm.coolwarm)
#plt.show()
#ims = []
def matplotlib_rotate():
    for i in range(20):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(30, 10)
        x = y = np.arange(0, 10, 1)
        X, Y = np.meshgrid(x, y)


        im=ax.plot_surface(X, Y, Z[0,X,Y,i], rstride=1, cstride=1, cmap=cm.coolwarm)
    #    ims.append([im])
        plt.savefig("figs/"+str(i)+".jpg")
        plt.close()
def init():
    plot(X, Y, model=None, title="", option={})
    return fig,

matplotlib_rotate()
#anim = animation.ArtistAnimation(fig,ims,interval=1000)
#fig.show()
