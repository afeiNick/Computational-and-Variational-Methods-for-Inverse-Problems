# Initialization
from dolfin import *
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import nb
from unconstrainedMinimization import InexactNewtonCG

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)


# Set the level of noise:
noise_std_dev = .3

# Load the image from file
data = np.loadtxt('logo.dat', delimiter=',')
np.random.seed(seed=1)
noise = noise_std_dev*np.random.randn(data.shape[0], data.shape[1])

# Set up the domain and the finite element space.
Lx = float(data.shape[1])/float(data.shape[0])
Ly = 1.0

mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), 140, 160)
V = FunctionSpace(mesh, "Lagrange",1)

# Generate the true image (u_true) and the noisy data (u_0)
class Image(Expression):
    def __init__(self, Lx, Ly, data):
        self.data = data
        self.hx = Lx/float(data.shape[1]-1)
        self.hy = Ly/float(data.shape[0]-1)
        
    def eval(self, values, x):
        j = math.floor(x[0]/self.hx)
        i = math.floor(x[1]/self.hy)
        values[0] = self.data[i,j]

trueImage = Image(Lx,Ly,data)
noisyImage = Image(Lx,Ly,data+noise)
u_true  = interpolate(trueImage, V)
u_0 = interpolate(noisyImage, V)

#nb.multi1_plot([u_true,u_0], titles=["True Image", "Noisy Image"])
#plt.show()

# functions that evaluate the true and the noisy image
u_true_fun = Function(V)
u_0_fun = Function(V)
u_true_fun.assign(u_true)
u_0_fun.assign(u_0)


uh = Function(V)
u_hat = TestFunction(V)
u_tilde = TrialFunction(V)

####### TN denoising #######
k_list = [1e-5, 1e-6, 1e-7]
uh_list = []
for k_val in k_list:
    k = Constant(k_val)

    uh.assign(interpolate(noisyImage, V))

    FTN = ( pow(uh-u_0_fun, 2) + Constant(0.5)*k*inner(nabla_grad(uh), nabla_grad(uh)) )*dx 
    parameters={"symmetric": True, "newton_solver": {"relative_tolerance": 1e-7,\
                                                    "report": True, \
                                                    "linear_solver": "cg",\
                                                    "preconditioner": "petsc_amg",\
                                                    "maximum_iterations": 100}}

    # first variation
    grad = (u_hat * Constant(2.0)*(uh - u_0_fun) + k*inner(nabla_grad(uh), nabla_grad(u_hat))) * dx
    H = (u_hat * Constant(2.0)*u_tilde + k*inner(nabla_grad(u_tilde), nabla_grad(u_hat))) * dx
                                                                                                                                                                                                                                                                                                                                                                                             
    solve(grad == 0, uh, J=H, solver_parameters=parameters)
    uh_list.append(uh)
    print "Built-in FEniCS solver."
    print "Norm of the gradient at converge", assemble(grad).norm("l2")
    print "Value of the energy functional at convergence", assemble(FTN)
    
#nb.multi1_plot([u_true,u_0,uh], titles=["True Image", "Noisy Image", "alpha = 0.1"])
nb.multi1_plot(uh_list, titles=["alpha = " + x for x in ["1e-5","1e-6","1e-7"]])
plt.show()


####### TV denosing #######
k = Constant(0.1)
eps_list = [10]
u_list = []
solver = InexactNewtonCG()
solver.parameters["rel_tolerance"] = 1e-5
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["gdu_tolerance"] = 1e-18
solver.parameters["max_iter"] = 1000
solver.parameters["c_armijo"] = 1e-5
solver.parameters["print_level"] = 1
solver.parameters["max_backtracking_iter"] = 10

for eps_val in eps_list:
    u = Function(V)
    u.assign(interpolate(noisyImage, V))
    eps = Constant(eps_val)
    
    FTV = ( pow(u-u_0_fun, 2) + k*sqrt(inner(nabla_grad(u), nabla_grad(u)) + eps ))*dx

    # first variation
    gradTV = (u_hat * Constant(2.0)*(u - u_0_fun) + k*inner(nabla_grad(u), nabla_grad(u_hat)) \
              /sqrt(inner(nabla_grad(u),nabla_grad(u)) + eps)) *dx
    HTV    = (u_hat * Constant(2.0)*u_tilde + k*((inner(nabla_grad(u),nabla_grad(u))+eps)*inner(nabla_grad(u_tilde), nabla_grad(u_hat))\
              - inner(nabla_grad(u), nabla_grad(u_tilde))*inner(nabla_grad(u), nabla_grad(u_hat))) \
              /pow(inner(nabla_grad(u),nabla_grad(u))+eps, 1.5) ) * dx        


    solver.solve(FTV, u, gradTV, HTV)
    u_list.append(u)

nb.multi1_plot(u_list, titles=["alpha = 0.1, eps = "+str(x) for x in eps_list])
plt.show()