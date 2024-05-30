import cmath
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def dft(f):
    N=int(len(f))
    mm=[]
    for k in range(N):
        F=0
        for n in range(N):
            v=2*np.pi*n*k*1j/N
            F=F+f[n]*np.exp(-v)
        mm.append(F/N)
    
    return mm



def dft_1(f):
    N=int(len(f))
    mm=[]
    for k in range(N):
        F=0
        for n in range(N):
            v=2*np.pi*n*k*1j/N
            F=F+f[n]*np.exp(v)
        mm.append(F)
    
    return mm
def f(t):
    return 2*np.sin(2*np.pi*t)+np.sqrt(abs(np.cos(2*np.pi*t)))

t1 = np.arange(0.0, 1.0, 0.01)
x=np.cos(2*np.pi*t1)
y=f(t1)

f1=[ ( x[i]+y[i]*1j ) for i in range(len(t1)) ]
f2=[ 4*np.exp(-np.pi*1j/2)*( x[i]+y[i]*1j ) for i in range(len(t1)) ]



F=dft(f1)
F_1=dft_1(F)
print(np.allclose(f1,F_1))#on verifie que dft_inverse de F est egal a f1
plt.figure()
plt.subplot(211)
plt.plot(f1,"b.",label="fonction initial")
plt.legend()
plt.subplot(212)
plt.plot(F_1,"r.",mfc='none',label="fonction apres dft et dft_inv")
plt.legend()
plt.show()



plt.figure()
plt.subplot(211)
plt.plot(np.cos(2*np.pi*t1), f(t1), 'b*',label="f1")
plt.legend()
plt.subplot(212)
plt.plot(t1,np.abs(F),"g.",label="spectre de f1")
plt.legend()
plt.show()



plt.figure()
plt.subplot(211)
plt.plot(np.cos(2*np.pi*t1), f(t1), 'b*',label="f1")
plt.ylabel("Y")
plt.xlabel("X")
plt.subplot(212)
plt.plot(np.real(f2), np.imag(f2), 'ro',label="f2")
plt.ylabel("Y")
plt.xlabel("X")

plt.show()
F2=dft(f2)
I=np.abs(F)/np.abs(F[2])
I2=np.abs(F2)/np.abs(F2[2])
plt.plot(I,"r*")
plt.plot(I2,"o",mfc='none',ms=10)
plt.show()