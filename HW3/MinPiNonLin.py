# Example: Nonlinear energy functional minimization
#
# In this example we solve the minimal surface problem
#
# Here the energy functional Pi(u) has the form
# Pi(u) = sqrt(1 + |\nabla u|^2) dx

# Load modules

from dolfin import *

import math
import numpy as np
import logging

import matplotlib.pyplot as plt
import nb

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

# Define the mesh and finite element spaces

nx = 32
ny = 32
mesh =  Mesh("circle.xml")
Vh = FunctionSpace(mesh, "CG", 1)

uh = Function(Vh)
u_hat = TestFunction(Vh)
u_tilde = TrialFunction(Vh)

nb.plot(mesh)
print "dim(Vh) = ", Vh.dim()

# Define the energy functional
Pi = sqrt(Constant(1.0) + inner(nabla_grad(uh), nabla_grad(uh)))*dx

def Dirichlet_boundary(x, on_boundary):
    if (x[0]*x[0] + x[1]*x[1]) > 0.99:
        return True
    else:
        return False

u_0 = Expression("pow(x[0], 4) - pow(x[1], 2)")
bc = DirichletBC(Vh, u_0, Dirichlet_boundary)
u_zero = Constant(0.)
bc_zero = DirichletBC(Vh,u_zero, Dirichlet_boundary)

# First variation (gradient)

grad = inner(nabla_grad(u_hat), nabla_grad(uh)/sqrt(Constant(1.0)+inner(nabla_grad(uh), nabla_grad(uh))))*dx

# Verify gradient expression
# This step can be skipped the first time you look at this

u0 = interpolate(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), Vh)
#u0 = interpolate(Expression("pow(x[0], 4) - pow(x[1], 2)"), Vh)

n_eps = 32
eps = 1e-2*np.power(2., -np.arange(n_eps))
err_grad = np.zeros(n_eps)

uh.assign(u0)
pi0 = assemble(Pi)
grad0 = assemble(grad)

dir = Function(Vh)
dir.vector().set_local(np.random.randn(Vh.dim()))
bc.apply(dir.vector())
dir_grad0 = grad0.inner(dir.vector())

for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector()) #uh = uh + eps[i]*dir
    piplus = assemble(Pi)
    err_grad[i] = abs( (piplus - pi0)/eps[i] - dir_grad0 )

plt.figure()
plt.loglog(eps, err_grad, "-ob")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(["Error Grad", "First Order"], "upper left")


# Second variation (Hessian)

denominator = pow(Constant(1.0) + inner(nabla_grad(uh), nabla_grad(uh)), 1.5)

                  
H = (inner(nabla_grad(u_tilde), nabla_grad(u_hat)) / sqrt(Constant(1.0) + inner(nabla_grad(uh), nabla_grad(uh))) \
  - inner(nabla_grad(uh), nabla_grad(u_hat)) * inner(nabla_grad(uh), nabla_grad(u_tilde)) / denominator) * dx

# Verify Hessian expression
# This step can be skipped the first time you look at this

uh.assign(u0)
H_0 = assemble(H)
err_H = np.zeros(n_eps)
for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector())
    grad_plus = assemble(grad)
    diff_grad = (grad_plus - grad0)
    diff_grad *= 1/eps[i]
    H_0dir = H_0 * dir.vector()
    err_H[i] = (diff_grad - H_0dir).norm("l2")

plt.figure()
plt.loglog(eps, err_H, "-ob")
plt.loglog(eps, (.5*err_H[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the second variation (Hessian)")
plt.xlabel("eps")
plt.ylabel("Error Hessian")
plt.legend(["Error Hessian", "First Order"], "upper left")


# Infinite-dimensional Newton Method

# Starting guess satisfies Dirichlet bdry condition
uh.assign(interpolate(u_0, Vh))

rtol = 1e-8
max_iter = 100

pi0 = assemble(Pi)
g0 = assemble(grad)
tol = g0.norm("l2")*rtol

du = Function(Vh)

lin_it = 0
print "{0:3} {1:3} {2:15} {3:15} {4:15}".format(
      "It", "cg_it", "Energy", "(g,du)", "||g||l2")

for i in range(max_iter):
    [Hn, gn] = assemble_system(H, grad, bc_zero)
    if gn.norm("l2") < tol:
        print "\nConverged in ", i, "Newton iterations and ", lin_it, "linear iterations."
        break
    myit = solve(Hn, du.vector(), gn, "cg", "petsc_amg")
    
    lin_it = lin_it + myit
    uh.vector().axpy(-1., du.vector())
    pi = assemble(Pi)
    print "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e}".format(
      i, myit, pi, -gn.inner(du.vector()), gn.norm("l2"))

plt.figure()
nb.plot(uh, mytitle="Solution")


# The built-in non-linear solver in FEniCS

uh.assign(interpolate(u_0, Vh))
parameters={"symmetric": True, "newton_solver": {"relative_tolerance": 1e-7,\
                                                 "report": True, \
                                                 "linear_solver": "cg",\
                                                 "preconditioner": "petsc_amg",\
                                                 "maximum_iterations": 100}}
                                                 
#solve(grad == 0, uh, bc, solver_parameters=parameters)
# If available, the Hessian/Jacobian can be specified
solve(grad == 0, uh, bc, J=H, solver_parameters=parameters)
print "Built-in FEniCS non linear solver."
print "Norm of the gradient at converge", assemble(grad).norm("l2")
print "Value of the energy functional at convergence", assemble(Pi)
plt.figure()
nb.plot(uh, mytitle="Build-in solver")
plt.show()