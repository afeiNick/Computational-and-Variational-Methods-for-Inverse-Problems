# -*- coding: utf-8 -*-
"""This program solves anisotropic Poisson problem

    - div (A(x,y) * grad u(x, y)) = f(x, y)

on the unit disc with source f given by

    f(x, y) = exp(-100*( pow(x[0], 2) + pow(x[1], 2) ))

and boundary conditions given by

    u(x, y) = 0
"""

from dolfin import *

A1 = Expression((("10", "0"), ("0", "10")))
A2 = Expression((("1", "-5"), ("-5", "100")))
# Create mesh and define function space
mesh =  Mesh("circle.xml")
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary
def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("exp(-100*( pow(x[0], 2) + pow(x[1], 2) ))")
a = inner(A1*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("A2.pvd")
file << u

# Plot solution
plot(u, interactive=True)
