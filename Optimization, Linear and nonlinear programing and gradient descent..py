"""

Optimization, Linear and nonlinear programing

l_infty, l_4 approximation, L1 approximation with trigonometric polynomials

Gradient descent for
bisection search, Logistc regression, Support vector machine, 
Stochastic robust approximation, Convolution with huber loss

@author: Kyungho Lee at SNU Econ

"""

import numpy as np
import numpy.linalg as LA
import cvxpy as cp

import matplotlib.pyplot as plt



"""

Data fitting with l_infty norm by linear programming

"""

# Data Assignment

import numpy as np
np.random.seed(0)
A = np.random.randn(30, 20)
b = np.random.randn(30)

# Curve fitting by LP

x = cp.Variable(20)
t = cp.Variable(1)
objective = cp.Minimize(t)
constr = []
constr += [-t*np.ones(30) <= A@x - b, A@x - b <= t*np.ones(30)]
prob = cp.Problem(objective,constr)
prob.solve(solver = cp.ECOS)

# Plotting

plt.plot(np.arange(30),b,'ro',label='b')
plt.plot(np.arange(30),A@x.value,'bx',label='fitted')
plt.legend()
plt.show()

"""

L_1 approximation of 2pi-periodic function f with trigonometric polynomials
by linear programming

"""

## Data assignment and grid

N=50
tt = np.linspace(-np.pi,np.pi,2*N+1)
yy = np.abs(tt) <= np.pi/2    
K=20
kk = np.array(range(K+1))
tt = np.reshape(np.repeat(tt,K+1),[2*N+1,K+1])
kt_a = tt*kk
kt_b = tt[:,1:]*kk[1:]

## LP

a = cp.Variable(K+1)
b = cp.Variable(K)
j = cp.Variable(2*N+1)
objective = cp.Minimize(sum(j)) # ignore np.pi/N that does not affect minimization
constr = []
constr += [-j <= (np.pi/N)*np.cos(kt_a)@a+np.sin(kt_b)@b-yy, (np.pi/N)*np.cos(kt_a)@a+np.sin(kt_b)@b-yy<=j]
prob = cp.Problem(objective, constr)
prob.solve(solver=cp.ECOS)
    
a = a.value
b = b.value
j= j.value

# Plotting Results

plt.figure()
plt.plot(tt[:,0],yy,label = "y")
plt.plot(tt[:,0],(np.pi/N)*np.cos(kt_a)@a+np.sin(kt_b)@b,label = "f(t)")
plt.legend()
plt.show()



"""

l_4 norm approximation as a convex QCQP

"""


m, n = 30, 20 
np.random.seed(0)
A = np.random.randn(m,n)
b = np.random.randn(m)

x = cp.Variable(n)
z = cp.Variable(m)

obj = cp.Minimize(sum(z**2))
constr = []
constr += [z >= (A@x - b)**2]

prob = cp.Problem(obj,constr)
prob.solve(solver = 'ECOS')

plt.title("Problem 4 l-4 norm")
plt.plot(np.arange(30),b,'ro',label='b')
plt.plot(np.arange(30),A@x.value,'bx',label='fitted')
plt.legend()
plt.show()



"""

Bisection for 1D convex minimization

"""

"""

Note on functions

f(x) = (1/2)*||Ax+b||^{2}
grad f(x) = A'Ax-A'b (nxm mxn nx1  nxm mx1)
h(alpha) = f(x - alpha grad f(x)) = (A(x-alpha*grad f(x))-b)'(A(x-alpha*grad f(x))-b)/2
h'(alpha) = grad f(x)'grad f(x-alpha * grad f(x))\
    
"""


# Generating Data
np.random.seed(0)
A = np.random.randn(30 ,20)
b = np.random.randn(30)
x = np.zeros(20)


# Number of iteration
N = 500
f_val = []  # list to contain obj function values
eps = 0.001  # Tolerance


for _ in range(N):
    grad_fx = A.T@(A@x-b)
    
    # Bisection for a linesearch
    
    l,u = -20, 5  # Reasonably sufficient range
    while u - l > eps:
        mid = (l+u)/2
        grad_fx_temp = A.T@(A@(x-mid*grad_fx)-b)  
        h_prime_alpha = grad_fx.T@grad_fx_temp  # needs to be zero.
        
        if h_prime_alpha <= 0:    
            u = mid
        else:
            l = mid
    
    # plug the argmin value 
    alpha_star = mid
    
    
    # Gradient Descent
    x -= alpha_star*grad_fx
    f_val.append(0.5*np.linalg.norm(A@x-b)**2)

x_opt = np.linalg.inv(A.T@A)@A.T@b
f_val_opt = 0.5*np.linalg.norm(A@x_opt-b)**2

plt.figure()
plt.plot(f_val,label="Linesearch")
plt.axhline(f_val_opt,label="Optimal f value",color='orange')
plt.legend()
plt.show()

# Check whether our solution x is a minimizer

print(x)
print(x_opt)  # We already know the explicit solution.

"""

Logistic Regression using SGD, Minibatch SGD with replacement, 
Minibatch SGD without replacement, Cyclic SGD, and Random Reshuing SGD

"""

# Generating Data

m, n = 30, 20 
np.random.seed(0)
A = np.random.randn(m,n)
y = 2*np.random.randint(2,size=m)-1

N = 30 # number of epochs

def Logistic_Regression(x,y,A):
    
    m, n = y.size, x.size
    y_rep = np.repeat(y, n).reshape((m,n))
    
    return np.mean(np.log(1+np.exp(-y_rep*A@x)))

def Logistic_Grad(x,y,A):

    m, n = y.size, x.size   
    y_rep = np.repeat(y, n).reshape((m,n))

    return -y_rep*A.T*np.exp(-y_rep*A@x)/(1 + np.exp(-y_rep*A@x))
    
## SGD
x = np.zeros(n)
alpha = 0.1

f_val_SGD = []
for _ in range(N*m):
    ind = np.random.randint(m)  # SGD
    grad_SGD = -y[ind]*A[ind,:].T*np.exp(-y[ind]*A[ind,:]@x)/(1 + np.exp(-y[ind]*A[ind,:]@x))
    x -= alpha*grad_SGD
    f_val_SGD_all = []
    f_val_SGD.append(Logistic_Regression(x,y,A))
    

##  Minibatch SGD with replacement
B = 10    #size of minibatch
x = np.zeros(n)
f_val_MB = []
for _ in range(N*m//B):  # // is integer division
    grad_MB = np.zeros(n)
    for _ in range(B):
        ind = np.random.randint(m)
        grad_MB += -y[ind]*A[ind,:].T*np.exp(-y[ind]*A[ind,:]@x)/(1+np.exp(-y[ind]*A[ind,:]@x))
    x -= alpha*grad_MB
    f_val_MB.append(Logistic_Regression(x,y,A))
    
##  Minibatch SGD without replacement
x = np.zeros(n) 
f_val_RP = []
for _ in range(N*m//B):  # // is integer division
    perm = np.random.permutation(np.arange(m))
    grad_RP = np.zeros(n)
    for j in range(B):
        ind = perm[j]
        grad_RP += -y[ind]*A[ind,:].T*np.exp(-y[ind]*A[ind,:]@x)/(1+np.exp(-y[ind]*A[ind,:]@x))
    x -= alpha*grad_RP
    f_val_RP.append(Logistic_Regression(x,y,A))

##  Cyclic SGD    
x = np.zeros(n)   
f_val_cyclic = []
for j in range(N*m):
    ind = j % m
    grad_cyclic = -y[ind]*A[ind,:].T*np.exp(-y[ind]*A[ind,:]@x)/(1+np.exp(-y[ind]*A[ind,:]@x))
    x -= alpha*grad_cyclic
    f_val_cyclic_all = []
    f_val_cyclic.append(Logistic_Regression(x,y,A))


## Random Shuffling SGD
x = np.zeros(n)
f_val_shuffle_cyclic = []
for j in range(N*m):
    if j%m == 0:
        perm = np.random.permutation(np.arange(m))
    ind = perm[j%m]
    grad_random_cyclic = -y[ind]*A[ind,:].T*np.exp(-y[ind]*A[ind,:]@x)/(1+np.exp(-y[ind]*A[ind,:]@x))
    x -= alpha*grad_random_cyclic
    f_val_shuffle_cyclic.append(Logistic_Regression(x,y,A))
    
    
plt.plot(np.arange(N*m)/m,f_val_SGD,'g')
plt.plot(np.arange(N*m//B)*B/m,f_val_MB,'k')
plt.plot(np.arange(N*m//B)*B/m,f_val_RP,'b')
plt.plot(np.arange(N*m)/m,f_val_cyclic,'orange')
plt.plot(np.arange(N*m)/m,f_val_shuffle_cyclic,'red')
plt.xlabel('Epochs')
plt.ylabel('f(x^k)')
plt.legend(['SGD','Mini-batch SGD with replacement','Mini-batch SGD without replacement','Cyclic SGD','Random Shuffle SGD'])
plt.title("Logistic Regression Optimization")
plt.show()


"""

Support Vector Machine using SGD

"""

alpha = 0.01
lmda = 0.1
x = np.zeros(n)

## SGD for SVM

def Support_Vector_Machine(x,y,A,lmda):
    m, n = y.size, x.size
    y_rep = np.repeat(y, n).reshape((m,n))
    max_value = np.max([np.array([0]*m), 1 - y_rep*A@x],axis=0)
    return np.mean(max_value) + lmda * (x.T@x)

f_val_SVM = []
for _ in range(N*m):
    ind = np.random.randint(m)
    if 1 > y[ind]*A[ind,:]@x:         
        grad_SVM_SGD = -y[ind]*A[ind,:].T + lmda*x
    elif 1 < y[ind]*A[ind,:]@x: 
        grad_SVM_SGD = lmda*x
    else:
        print("Error")
    x -= alpha*grad_SVM_SGD
    f_val_SVM.append(Support_Vector_Machine(x,y,A,lmda))
    
plt.plot(np.arange(N*m)/m,f_val_SVM,'k')
plt.xlabel('Epochs')
plt.ylabel('f(x^k)')
plt.legend(['SGD'])
plt.title("Support Vector Machine Optimization")
plt.show()


"""

Stochastic Robust Approximation using SGD

"""

import numpy as np

m, n = 30, 20 
np.random.seed(0)
Abar = np.random.randn(m,n) 
b = np.random.randn(m)
Sigma_root = 0.01*np.random.randn(m*n,m*n) 
alpha = 0.001

def generate_A():
    return Abar + (Sigma_root@np.random.randn(m*n)).reshape((m,n))

N = 30 # number of epochs
 

def huber_loss(x) :
    return np.sum((1/2)*(x**2)*(np.abs(x)<=1) + (np.sign(x)*x-1/2)*(np.abs(x)>1) )
def huber_grad(x) :
    return x*(np.abs(x)<=1) + np.sign(x)*(np.abs(x)>1)

x = np.zeros(n)

Huber_SGD = []
for _ in range(N*m):
    A_temp = generate_A()
    grad_SGD = huber_grad(A_temp@x-b)@A_temp
    x -= alpha*grad_SGD
    apx_obj = np.mean([huber_loss(generate_A()@x-b) for _ in range(100)])
    Huber_SGD.append(apx_obj)
    
plt.plot(np.arange(N*m)/m,Huber_SGD,'k')
plt.xlabel('Epochs')
plt.ylabel('f(x^k)')
plt.legend(['SGD'])
plt.title("Stochastic Robust Approximation")
plt.show()

"""

Implementation of Convolution Huber Loss SGD through Duck Typing

"""

class Convolution1d :
    
    ''' Inite object with filter'''
    def __init__(self, filt) :
        self.__filt = filt
        self.__r = filt.size
        self.T = TransposedConvolution1d(self.__filt)

    '''
    Convolution operation
        Usage : self @ vector
    '''
    def __matmul__(self, vector) :
        r, n = self.__r, vector.size
        
        return np.asarray([np.sum(self.__filt*vector[i:i+r]) for i in np.arange(n-r+1)])  # IMPLEMENT THIS
    
'''
Transpose of 1-dimensional convolution operator used for the 
transpose-convolution operation A.T@(...)
'''
class TransposedConvolution1d : 
    
    ''' Init TransposedConvolution1d object with filter '''
    def __init__(self, filt) :
        self.__filt = filt
        self.__r = filt.size

    '''
    Convolution operation
        Usage : A @ vector
    '''
    def __matmul__(self, vector) :
        r = self.__r
        n = vector.size + r - 1
        return np.asarray([np.sum(np.flip(self.__filt)*np.asarray([0]*(r-1) + list(vector) + [0]*(r-1))[i:i+r]) for i in np.arange(n)])  # IMPLEMENT THIS

# Test Code
def huber_loss(x) :
    return np.sum( (1/2)*(x**2)*(np.abs(x)<=1) + (np.sign(x)*x-1/2)*(np.abs(x)>1) )
def huber_grad(x) :
    return x*(np.abs(x)<=1) + np.sign(x)*(np.abs(x)>1)

r, n, lam = 6, 20, 0.1

np.random.seed(0)
k = np.random.randn(r)
b = np.random.randn(n-r+1)
x = np.ones(n)
x_alt = x

A = Convolution1d(k)

## For Comparison
from scipy.linalg import circulant
A_alt = circulant(np.concatenate((np.flip(k),np.zeros(n-r))))[r-1:,:]

# SGD Implementation

alpha = 0.001
x_list = []
f_val_list = []
f_val_alt_list = []
for _ in range(100) :
    x = x - alpha*(A.T@(huber_grad(A@x-b))+lam*x)
    x_alt = x_alt - alpha*(A_alt.T@(huber_grad(A_alt@x_alt-b))+lam*x_alt)
    f_val_list.append(huber_loss(A@x-b)+0.5*np.linalg.norm(x*x)**2)
    f_val_alt_list.append(huber_loss(A_alt@x_alt-b)+0.5*np.linalg.norm(x_alt*x_alt)**2)

plt.plot(np.arange(100),f_val_list,'g')
plt.plot(np.arange(100),f_val_alt_list,'k')
plt.xlabel('Epochs')
plt.ylabel('f(x^k)')
plt.legend(['Convlution1d','Numpy Circulant'])
plt.title("Convolution Huber Loss SGD")
plt.show()

