"""

@author: Kyungho Lee at SNU Econ

Note: Some starter codes are provided by the lecturer.
"""

"""

PD-controller with delays.

"""

## class Drone1D was provided by the lecturer.


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Drone1D :
    '''
        Drone object
        controller : controller function
        mass : mass in kilogram(kg)
        init_height : initial height in meters(m)
        init_velocity : initial velocity in meters per second(m/s)
    '''
    def __init__(self, controller, mass, init_height, init_velocity, d) :
        self.__mass = mass                  # in kilograms (mass remains constant)
        self.__height = init_height         # initial height in meters
        self.__velocity = init_velocity     # initial velocity in m/s
        self.__controller = controller      # controller function
        self.__d = d
    '''
    Update next state
        height_history, velocity_history : list of height and list of velocity
        curr_power : thruster power which will update the state
    '''
    def __next_state_function(self, height_history, velocity_history, curr_power) :
        # drone mass, gravitational acceleration, time interval length
        m, g, dt = self.__mass, 9.80665, 0.1
        
        curr_height = height_history[-1]
        curr_velocity = velocity_history[-1]

        next_height = curr_height + dt*curr_velocity
        next_velocity = curr_velocity + (dt/m)*(curr_power - m*g)
        
        height_history.append(next_height)
        velocity_history.append(next_velocity)


    
    # Simulation method
    ''' Simulate drone for terminal_time * dt seconds(sec) '''
    def simulation(self, noise_function, terminal_time) :
        # Initialize history
        height_history = [self.__height]
        velocity_history = [self.__velocity]
        d = self.__d
        # Simulation loop
        for _ in range(terminal_time) :
            curr_power = self.__controller(height_history, velocity_history,d)
            curr_noise = noise_function(height_history, velocity_history)
            self.__next_state_function(height_history, velocity_history, curr_power+curr_noise)

        # Simulation video
        self.video(height_history)


    ''' Create a simulation video '''
    def video(self, height_history) :
        dt = 0.1

        t = len(height_history)
        
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2)
        
        # Plot the graph of height
        h_graph = fig.add_subplot(gs[0, 1])
        h_graph.plot(np.arange(t)*dt, height_history, 'r-')
        h_graph.set_xlabel('Time(s)')
        h_graph.set_ylabel('Height(m)')
        
        
        # Set size of video window
        simu = fig.add_subplot(gs[0, 0])
        simu.set_ylabel("height (m)")
        simu.grid()
        simu.set_xlim(-1.0, 1.0)
        h_max = np.amax(height_history)
        h_min = np.amin(height_history)
        h_mean = (h_max + h_min)/ 2
        simu.set_ylim(-3,3)
        
        # Plot drone as a square
        h_data = height_history
        drone, = simu.plot(0.0, h_data[0], marker=(2, 0, 90), markersize=50, markeredgewidth=10, color = 'k')
        time_text = simu.text(0.05, 0.8, '', transform=simu.transAxes)
        time_template  = 'Current time = %.1f$s$\n'
        time_template += 'height = %.2f$m$\n'
        
        def animate(frame) :
            drone.set_data(0.0, h_data[frame])
            time_text.set_text(time_template % (frame/10, h_data[frame]))
            return drone, time_text
            
        # interval : delay between frames in milliseconds
        video = FuncAnimation(fig, func=animate, frames=np.arange(0, t), interval=10*dt)
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        plt.show()

def noise_function(height_history, velocity_history) :
    return 0
#     noise_level = 0.1
#     return np.random.normal(0, noise_level)


def PD_controller_with_delays(height_history, velocity_history,d) :
    mass, g = 10, 9.80665  #PD controller requires knowledge of mass
    alpha, beta = 2.0, 1.0
    height_history = list(np.zeros(d)) + height_history
    velocity_history = list(np.zeros(d)) + velocity_history
    return mass * g - alpha * height_history[-d] - beta * velocity_history[-d]

Drone1D(PD_controller_with_delays, mass = 10, init_height = 0.5, init_velocity = 1, d=4).simulation(noise_function, 1000)

import numpy as np
import scipy.sparse as ss

mass, g = 10, 9.80665  #PD controller requires knowledge of mass
alpha, beta = 2.0, 1.0
h=0.1
d= 3

first_row = [1,h] + [0,0]*d
second_row = [0,1]+[0,0]*(d-1)+[-alpha*h/mass, -beta*h/mass]

two_rows = np.array([first_row,second_row])
Block_I = ss.diags([1],[-2],shape=(2*(d+1),2*(d+1)))
Block_I = Block_I.toarray()

A = np.concatenate([two_rows,Block_I[2:,]])
print("Spectral radius is: {0:0.5f}".format(max(np.abs(np.linalg.eigvals(np.array(A).astype(np.float64))))))

# It is guessed that the staibility would hold until d < 4


"""

Knapsack Problem

"""

# Define KnapSack function

def KnapSack(v,w,n,W):
    
    w = [0] + w
    v = [0] + v
    V = np.zeros((n+1,W+1))
    
    for i in np.arange(1,n+1):
        for weight in np.arange(W+1):
            if w[i] <= weight:
                V[i,weight] = np.max([V[i-1,weight] , v[i]+V[i-1,weight-w[i]]]) # Here we use recursive reltionship.
            else: # No need to consider 
                V[i,weight] = V[i-1,weight]
    return V[n,W]

## Setting

n = 15
v = [4, 3, 5, 7, 1, 9, 6, 9, 4, 5, 1, 2, 7, 4, 9]
w = [10, 2, 6, 8, 10, 3, 5, 5, 8, 9, 2, 2, 4, 1, 6]
W = 35

KnapSack(v,w,n,W)

# Keep tracking

def KnapSack_KeepTrack(v,w,n,W):
    
    w = [0] + w  # We need to iterate it from i = 1
    v = [0] + v  # We need to iterate it from i = 1
    V = np.zeros((n+1,W+1)) # Define m(0,w) = 0 -> We fill in V from i = 1
    keep = np.zeros(((n+1,W+1)))
    items = []
    
    for i in np.arange(1,n+1):
        for weight in np.arange(W+1):
            if (w[i] <= weight) & (V[i-1,weight] < v[i]+V[i-1,weight-w[i]]):
                V[i,weight] =  v[i]+V[i-1,weight-w[i]]
                keep[i,weight] = 1 # Record an used item for given weight.
            else:
                V[i,weight] = V[i-1,weight]
                keep[i,weight] = 0
    
    # Make a container for used items.
    
    K = W # Starts from the backwards
    
    for i in np.arange(n,0,-1):  # From the terminal time
        if keep[i,K] == 1:
            items.append(i)
            K = K - w[i]
    
    return V[n,W], np.array(items)

# Result:

result = KnapSack_KeepTrack(v,w,n,W)

# Check whther 'keep' works well.

sum(np.array(v)[list(result[1]-1)])
sum(np.array(w)[list(result[1]-1)])



## Test

v = [10, 40, 30, 50]
w = [5, 4, 6, 3]
W = 10
n = 4
result = KnapSack_KeepTrack(v,w,n,W)
KnapSack(v,w,n,W)

"""

Optimal disposition of a stock

log(p_i) ~ normal(mu_i, sigma_i**2)

The goal is to maximize the total expected revenue from the sales in the two rounds.

We consider three different information patterns.
 Prescient. You know p0 and p1 before you decide the amounts to sell in each period.
 No knowledge. You do not know the prices.
 Partial knowledge. You are told the price p0 before you decide how much to sell in period
0, and you are told the price p1 before you decide how much to sell in period 1.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For plotting a distribution


# Setting Parameters

B = 10
mu0 = 0
mu1 = 0.1
sigma0 = 0.4
sigma1 = 0.4

# Monte-Carlo

sample_size = 10000  
p0 = np.random.lognormal(mean = mu0, sigma = sigma0, size = sample_size)
p1 = np.random.lognormal(mean = mu1, sigma = sigma1, size = sample_size)

# We know explicit expectation
E_p0 = np.exp(mu0 + (sigma0**2)/2)
E_p1 = np.exp(mu1 + (sigma1**2)/2)


# 1 Prescient Case
 
s0_Prescient = np.zeros(sample_size)
s1_Prescient = np.zeros(sample_size)

s0_Prescient[p0 > p1] = B
s1_Prescient[p1 >= p0] = B

Revenue_Prescient = p0*s0_Prescient + p1*s1_Prescient


# 2 No Knowledge Case
# Only Sell at period 1 since exp(mu0+sigma0**2/2) < exp(mu1+sigma1**2/2) 

s0_No_Knowledge = np.zeros(sample_size)
s1_No_Knowledge = np.zeros(sample_size)
s1_No_Knowledge = B
Revenue_No_Knowledge = p0*s0_No_Knowledge  + p1*s1_No_Knowledge 

# Partial Knowledge

s0_Partial = np.zeros(sample_size)
s1_Partial = np.zeros(sample_size)
s0_Partial[p0 > E_p1] = B
s1_Partial[p0 <= E_p1] = B
Revenue_Partial = p0*s0_Partial  + p1*s1_Partial


plt.rc("ggplot2")
plt.figure()
plt.axvline(x = np.mean(Revenue_Prescient))
sns.distplot(Revenue_Prescient, label="Prescient",kde = False)
plt.axvline(x = np.mean(Revenue_No_Knowledge))
sns.distplot(Revenue_No_Knowledge, label="No Knowledge",kde = False)
plt.axvline(x = np.mean(Revenue_Partial))
sns.distplot(Revenue_Partial, label="Partial Knowledge",kde = False)
plt.legend()
plt.show()

# Print Expected Revenue

Info_list = ["Prescient", "No Knowledge", "Partial Knowledge"]
Reve_list = [Revenue_Prescient, Revenue_No_Knowledge, Revenue_Partial]

for i in range(3):
    print("Expected Revenue of {} Case is {:0.2f}".format(Info_list[i],np.mean(Reve_list[i])))


"""

Refined stochastic inventory control problem.

"""


# Define Functions to be used.
    
def g_store(x, s_lin=0.1, s_quad = 0.05):       
    return s_lin*x + s_quad*(x**2)

def g_rev(x, u, w, p_rev = 3):
    return -p_rev*np.min([x+u,w])

def g_unmet(x, u, w, p_unmet = 3):        
    return p_unmet*np.max([-x-u+w,0])

def g_sal(x, p_sal=1.5):
    return -p_sal*x

def g_order(u, u_disc = 6, p_fixed = 4, p_whole = 2, p_disc = 1.6):
       
    if u == 0:
        return 0
    elif 1<= u <= u_disc:
        return p_fixed + p_whole*u
    elif u > u_disc:
        return p_fixed + p_whole*u_disc + p_disc*(u-u_disc)
    
def next_state(x,u,w):
    return np.max([x + u - w,0])
    
def cost(t, x, u, w):
    if t < T:        
        g_order_t = g_order(u)
        g_store_t = g_store(x)
        g_rev_t = g_rev(x,u,w)
        g_unmet_t = g_unmet(x,u,w)        
        return g_order_t + g_store_t + g_rev_t + g_unmet_t
    
    # Consider the Terminal Cost.
    
    elif t == T:
        g_sal_T = g_sal(x)
        return g_sal_T

def Cost_To_Go(t, x, Cost_Matrix, C = 20):
    
    # Make a container for optimal average cost
    
    Expect_cost_List = []
    
    # Check whether we are at the Terminal time T
    
    if t == T:
        g_sal_T = g_sal(x)
        u_optimal = 0
        opt_cost_t = g_sal_T
    else:       
        
        # if not, use Bellamn function to append optimal average cost.
        
        for u in np.arange(0,C-x+1):
            
            # Since we know pmf of w, we can explicitly calculate average cost.
            # Use Cost_Matrix to pull out optimal average cost that was calculated before.
            
            Expect_cost_t = 0.2*(cost(t, x, u, w=0) + Cost_Matrix[t+1,next_state(x,u,w=0)]) + \
                        0.25*(cost(t, x, u, w=1) + Cost_Matrix[t+1,next_state(x,u,w=1)]) + \
                        0.25*(cost(t, x, u, w=2) + Cost_Matrix[t+1,next_state(x,u,w=2)]) + \
                        0.2*(cost(t, x, u, w=3) + Cost_Matrix[t+1,next_state(x,u,w=3)]) + \
                        0.1*(cost(t, x, u, w=4) + Cost_Matrix[t+1,next_state(x,u,w=4)])            

            
            Expect_cost_List.append(Expect_cost_t)  # Add calculated average cost
        
        u_optimal = np.argmin(Expect_cost_List)  #  Append Optimal action for a given state x
        opt_cost_t = np.min(Expect_cost_List)   # Optimal Cost
    
    return u_optimal, opt_cost_t

T = 50  # Time periods

C = 20  # Capacity

# Initialize Cost-to-go and controls.
Optimal_Cost_To_Go = np.zeros(shape=(T+1,C+1))
Optimal_Policy = np.zeros(shape=(T+1,C+1))


# Dynamic Programming: Starts at the Terminal time T
for t in np.arange(T,-1,-1): # Iterate over {0,1,...,T}
    for x in np.arange(C+1):  # At time t, we iterate over set of possible states x = {0,1,...,C}
        u_optimal, opt_cost_t = Cost_To_Go(t,x,Optimal_Cost_To_Go)
        Optimal_Cost_To_Go[t,x] = opt_cost_t
        Optimal_Policy[t,x] =  u_optimal

# Print the Optimal Cost

print("The Optimal Cost is {:0.2f}".format(Cost_To_Go(0,10,Optimal_Cost_To_Go)[1])) # Optimal Cost

# plotting

plt.rc('ggplot2')
plt.figure()
x_list = [x for x in range(C+1)]
for t in range(46,50):
    plt.plot(x_list,Optimal_Policy[t,:],label= 't = {}'.format(t))
plt.plot(x_list,Optimal_Policy[0,:],label= 't = {}'.format(1))
plt.legend()
plt.title("Refined Inventory Control: Optimal Policy")
plt.ylim(0,20)
plt.xlim(0,20)
plt.ylabel("Optimal Policy")
plt.xlabel("State: x")
plt.show()


"""

Notes on convergence of the Optimal Policy

Optimal Policy Converges as t goes to zero
from t = 47 (i.e. t <= 47) the Optimal Policy is 
Ordering 7 units if the state x = 0
Ordering 5 units if the state x = 1
Do not order if the state x >= 2

-> This is the steady state Optimal Policy

"""


"""

Bi-directional supply chain via LQR

"""

# Setting Data

n = 4
rho = 0.1
sigma = 1

A = np.eye(n)
B = np.zeros((n,n))
for i in np.arange(n):
     B[i,i] = 1
     if i < n-1:
         B[i,i+1] = -1
                  
print(np.linalg.matrix_rank(B)) # Check whether LDS is controllable


# Set up steady-state LQR controller

I = np.eye(n)
P = I 
for _ in range(10000):  # Iterate Enough Number of Time
    P = I + P - P@B@np.linalg.inv(rho*I+B.T@P@B)@B.T@P
K = -np.linalg.inv(rho*I+B.T@P@B)@B.T@P


"""

The Opitmal Policy: mu(x) = K@x

"""

# Optional
# The optimal cost is

# W = np.zeros((n,n))
# W[n-1,n-1] = 1

# J = np.sum(np.diag(P@W))


import numpy as np
import sympy as sp


"""

Characterizing Dynamical System of the differential equation

"""

d = sp.symbols('d')
m = sp.symbols('m')
k = sp.symbols('k')

A_sp = sp.Matrix([[0,0,0,0,1,0,0,0],\
               [0,0,0,0,0,1,0,0],\
               [0,0,0,0,0,0,1,0],\
               [0,0,0,0,0,0,0,1],\
               [-2*k,k,0,0,-2*d,d,0,0],\
               [k,-2*k,k,0,d,-2*d,d,0],\
               [0,k,-2*k,k,0,d,-2*d,d],\
                [0,0,k,-k,0,0,d,-d]])

# B_sp = sp.Matrix([[0,0,0,0,0,0,1,0]])

d = 0.01
m = 1
k = 1

A = np.array([[0,0,0,0,1,0,0,0],\
               [0,0,0,0,0,1,0,0],\
               [0,0,0,0,0,0,1,0],\
               [0,0,0,0,0,0,0,1],\
               [-2,1,0,0,-2*0.01,0.01,0,0],\
               [1,-2,1,0,0.01,-2*0.01,0.01,0],\
               [0,1,-2*1,1,0,0.01,-2*0.01,0.01],\
                [0,0,1,-1,0,0,0.01,-0.01]])

B = np.array([[0,0,0,0,0,0,1,0]])

B.shape
A.shape
n = np.size(A,0)

C = [B.T]
for i in range(n-1):
    C.append(A@C[i])    
C = np.concatenate(C,axis=1)
print(C)
print(np.linalg.matrix_rank(C)) # Because Rank is 6, it is not "Controllable".


# Case 1 acculator to the first


A_1 = np.array([[0,0,0,0,1,0,0,0],\
               [0,0,0,0,0,1,0,0],\
               [0,0,0,0,0,0,1,0],\
               [0,0,0,0,0,0,0,1],\
               [-2,1,0,0,-2*0.01,0.01,0,0],\
               [1,-2,1,0,0.01,-2*0.01,0.01,0],\
               [0,1,-2*1,1,0,0.01,-2*0.01,0.01],\
                [0,0,1,-1,0,0,0.01,-0.01]])

B_1 = np.array([[0,0,0,0,1,0,0,0]])

n = np.size(A_1,0)

C_1 = [B_1.T]
for i in range(n-1):
    C_1.append(A_1@C_1[i])    
C_1 = np.concatenate(C_1,axis=1)
print(C_1)
print(np.linalg.matrix_rank(C_1)) # Because Rank is 8, it is "Controllable".


"""

Ordering Matrix multiplication: Dynamic programming for optimal matrix multiplication.

"""

N = 15
n = [28 ,12 ,25 ,18 ,11 ,10 ,14 ,28 ,25 ,21 ,20 ,10 ,18 ,25 ,16 ,23]

def Matrix_Chain(n,N):
    
    """
    N : the number of matrices
    n : dimensions of matrices
    """
    # s = np.zeros((N+1,N+1))
    
    m = np.zeros((N+1,N+1)) # Initialize minimum operation time
        
    for l in range(2,N+1): # First, iterate over the fisrt step i.e. M_1 ... M_N
        for i in range(1,N-l+2):  # Pick where to split at first
            j = i + l - 1   # possible range of j
            m[i,j] = np.inf  # for the minimum comparison, we first let np.inf
            for k in range(i,j):  # iterate over i to j and find the optimal split k
                q = m[i,k] + m[k+1,j] + n[i-1]*n[k]*n[j]  # Use recursive relationship
                if q < m[i,j]:  
                    m[i,j] = q  # if we iterate k from i to j, we find the optimal cost m(i,j)
                    # s[i,j] = k
    return m[1:,1:]  # Cut-off unnecessary parts

result = Matrix_Chain(n,N)

# so the problem is solved as:
    
print(result[0,N-1])

"""

Infinite time horizon LQR.

Stochastic Inventory Model with delayed Demand.

"""


n,m = 1,4
rho = 0.5
sigma = 1

A = np.eye(n)
B = np.zeros((n,m))
B[0,m-1] = 1

Q = np.eye(n)
R = rho*np.eye(m)
P = Q

# for i in np.arange(n):
#      B[i,i] = 1
#      if i < n-1:
#          B[i,i+1] = -1
                  
print(np.linalg.matrix_rank(B)) # Check whether LDS is controllable

# Set up steady-state LQR controller

I = np.eye(n)
P = I 
for _ in range(10000):  # Iterate Enough Number of Time
    P = Q + P - P@B@np.linalg.inv(R+B.T@P@B)@B.T@P
K = -np.linalg.inv(R+B.T@P@B)@B.T@P

# Or we can use discrete_are solver

import scipy.linalg
P_dare = scipy.linalg.solve_discrete_are(A, B, Q, R)


"""

The Opitmal Policy: mu(x) = K@x

"""

# The optimal cost is

W = 1 # sigma
J = np.sum(np.diag(P*W))
J_dare = np.sum(np.diag(P_dare*W))

print(J)
print(J_dare)
