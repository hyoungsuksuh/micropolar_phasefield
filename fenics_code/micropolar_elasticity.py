# ---------------------------------------------------------------- 
# FEniCS implementation: Micropolar elasticity    
# Written by: Hyoung Suk Suh (h.suh@columbia.edu)     
# ----------------------------------------------------------------       

from dolfin import *
import sys
import time


tic = time.time()



# ---------------------------------------------------------------- 
# Input parameters
# ----------------------------------------------------------------
# Mesh and result file names
file_name = 'plate_with_a_hole'  # Input/output directory name

# Material parameters (micropolar elasticity)
G  = 50.0e9  # Shear modulus [Pa]
nu = 0.3     # Poisson's ratio
l  = 2.0     # Characteristic length [m]
N  = 0.9     # Coupling number 

# Applied traction
trac = 1.0e3 # traction [Pa]





# ---------------------------------------------------------------- 
# Read mesh
# ----------------------------------------------------------------
mesh = Mesh('./'+file_name+'/'+file_name+'.xml')

dim = mesh.geometric_dimension()
mesh_coord = mesh.coordinates()
mesh_xmin  = min(mesh_coord[:,0])
mesh_xmax  = max(mesh_coord[:,0])
mesh_ymin  = min(mesh_coord[:,1])
mesh_ymax  = max(mesh_coord[:,1])





# ---------------------------------------------------------------- 
# Define function spaces
# ----------------------------------------------------------------
u_elem     = VectorElement('CG', mesh.ufl_cell(), 2) # displacement
theta_elem = FiniteElement('CG', mesh.ufl_cell(), 1) # rotation

mixedUT = u_elem*theta_elem

V = FunctionSpace(mesh, mixedUT)

U, T     = V.split()
U_0, U_1 = U.split() # 2 dofs for disp. (U_0, U_1) / 1 dof for rotation (T)





# ---------------------------------------------------------------- 
# Define boundary conditions
# ----------------------------------------------------------------
top    = CompiledSubDomain("near(x[1], mesh_ymax) && on_boundary", mesh_ymax = mesh_ymax)
bottom = CompiledSubDomain("near(x[1], mesh_ymin) && on_boundary", mesh_ymin = mesh_ymin)
left   = CompiledSubDomain("near(x[0], mesh_xmin) && on_boundary", mesh_xmin = mesh_xmin)
right  = CompiledSubDomain("near(x[0], mesh_xmax) && on_boundary", mesh_xmax = mesh_xmax)

# Dirichlet boundary
BC_left         = DirichletBC(U_0, Constant(0.0), left)
BC_bottom       = DirichletBC(U_1, Constant(0.0), bottom)

BC_left_theta   = DirichletBC(T, Constant(0.0), left)
BC_bottom_theta = DirichletBC(T, Constant(0.0), bottom)

BC = [BC_left, BC_bottom, BC_left_theta, BC_bottom_theta]

# Neumann boundary
boundaries = MeshFunction('size_t', mesh, dim-1)
boundaries.set_all(0)
right.mark(boundaries,1)
traction = Constant((trac, 0.0))

ds = Measure("ds")(subdomain_data=boundaries)
n  = FacetNormal(mesh)





# ---------------------------------------------------------------- 
# Define variables
# ----------------------------------------------------------------
# Micropolar elastic material parameters -- conversion
lamda = G*((2.*nu)/(1.-2.*nu))
mu    = G*((1.-2.*N**2)/(1.-N**2))
kappa = G*((2.*N**2)/(1.-N**2))
gamma = 4.*G*l**2

# Micropolar strain & micro-curvature -------------
def epsilon(u, theta):
  
  strain = as_tensor([[ u[0].dx(0),         u[1].dx(0) - theta ],
                      [ u[0].dx(1) + theta, u[1].dx(1)         ]])  

  return strain
  

def epsilon_sym(u):
  
  strain_sym = as_tensor([[ u[0].dx(0),                        (1./2.)*(u[0].dx(1) + u[1].dx(0)) ],
                          [ (1./2.)*(u[0].dx(1) + u[1].dx(0)), u[1].dx(1)                        ]])
  
  return strain_sym


def epsilon_skew(u, theta):
  
  strain_skew  = as_tensor([[ 0.0,                                       (1./2.)*(u[1].dx(0) - u[0].dx(1)) - theta ],
                            [ (1./2.)*(u[0].dx(1) - u[1].dx(0)) + theta, 0.0                                       ]])
  
  return strain_skew


def phi(theta):
  
  curvature = as_vector([ theta.dx(0),
                          theta.dx(1) ]) 
  return curvature
# -------------------------------------------------


# Force stress & couple stress --------------------
def sigma_B(u):

  eps_sym = epsilon_sym(u)
  
  stress_B = lamda*tr(eps_sym)*Identity(2) + (2.*mu+kappa)*eps_sym

  return stress_B


def sigma_C(u, theta):
  
  eps_skew = epsilon_skew(u, theta)

  stress_C = kappa*eps_skew

  return stress_C


def sigma(u, theta):

  stress = sigma_B(u) + sigma_C(u, theta)
  
  return stress


def E3_sigma_C(u, theta):
  
  stress_C = sigma_C(u, theta)
  
  return stress_C[0,1] - stress_C[1,0]
  

def m_R(theta):
  
  curvature = phi(theta)
  
  couple = gamma*curvature

  return couple
# -------------------------------------------------




  
# ---------------------------------------------------------------- 
# Define variational form
# ----------------------------------------------------------------
# Define test & trial spaces
u,  theta = TrialFunctions(V)
eta, zeta = TestFunctions(V)

# Bilinear form
a = inner(epsilon(eta,zeta), sigma(u, theta)) * dx \
  + inner(phi(zeta), m_R(theta)) * dx              \
  - inner(zeta, E3_sigma_C(u, theta)) * dx
  
L = dot(eta, traction) * ds(1)

# Solution
x_h = Function(V)

# Define variational problem & solver
problem = LinearVariationalProblem(a, L, x_h, BC)
solver  = LinearVariationalSolver(problem)





# ---------------------------------------------------------------- 
# Solve system
# ----------------------------------------------------------------
solver.solve()
u_h, theta_h = x_h.split()





# ---------------------------------------------------------------- 
# Output results
# ----------------------------------------------------------------
# Field variables
vtkfile_u = File('./'+file_name+'/'+file_name+'_u.pvd')
vtkfile_u << u_h

vtkfile_theta = File('./'+file_name+'/'+file_name+'_theta.pvd')
vtkfile_theta << theta_h





toc = time.time() - tic

print('Elapsed CPU time: ', toc, '[sec]')