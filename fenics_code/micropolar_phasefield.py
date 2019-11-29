# ---------------------------------------------------------------- 
# FEniCS implementation: Micropolar phase field fracture   
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
file_name = 'double_notch'  # Input/output directory name
degradation = 'B+C+R'       # Energy parts to be degraded: B, C, R, B+R, B+C, B+C+R

# Material parameters 1 (micropolar elasticity)
G  = 12.5e3    # Shear modulus [MPa]
nu = 0.2       # Poisson's ratio
l  = 30.0      # Characteristic length (bending) [mm]
N  = 0.5       # Coupling parameter 

# Material parameters 2 (phase field fracture)
Gc     = 0.1     # Critical energy release rate [N/mm]
lc     = 0.75    # Length scale [mm]
psi_cr = 0.001   # Threshold strain energy per unit volume [MJ/m3]
p      = 10.0    # Shape parameter

# Solver parameters
t_i       = 0.0     # Initial t [sec]
t_f       = 0.1     # Final t [sec]
dt        = 0.0005  # dt [sec]
disp_rate = 1.0     # Displacement rate [mm/s]

staggered_tol     = 1e-6 # tolerance for the staggered scheme
staggered_maxiter = 10   # max. iteration for the staggered scheme
newton_Rtol       = 1e-8 # relative tolerance for Newton solver (balance eq.)
newton_Atol       = 1e-8 # absoulte tolerance for Newton solver (balance eq.)
newton_maxiter    = 20   # max. iteration for Newton solver (balance eq.)
snes_Rtol         = 1e-9 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 30   # max. iteration for SNEs solver (phase field eq.)





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
u_elem     = VectorElement('Lagrange', mesh.ufl_cell(), 2) # displacement
theta_elem = FiniteElement('Lagrange', mesh.ufl_cell(), 1) # rotation

mixedUT = u_elem*theta_elem

V = FunctionSpace(mesh, mixedUT)

U, T     = V.split()
U_0, U_1 = U.split() # 2 dofs for disp. (U_0, U_1) / 1 dof for rotation (T)

# function spaces for the phase field, history variable
W  = FunctionSpace(mesh, 'CG', 1) # phase field
WW = FunctionSpace(mesh, 'DG', 0) # history variable





# ---------------------------------------------------------------- 
# Define boundary conditions
# ----------------------------------------------------------------
top    = CompiledSubDomain("near(x[1], mesh_ymax) && on_boundary", mesh_ymax = mesh_ymax)
bottom = CompiledSubDomain("near(x[1], mesh_ymin) && on_boundary", mesh_ymin = mesh_ymin)
left   = CompiledSubDomain("near(x[0], mesh_xmin) && on_boundary", mesh_xmin = mesh_xmin)
right  = CompiledSubDomain("near(x[0], mesh_xmax) && on_boundary", mesh_xmax = mesh_xmax)

# constrained displacement boundary
BC_bottom     = DirichletBC(U, Constant((0.0,0.0)), bottom)

# constrained micro-rotation boundary
BC_top_theta    = DirichletBC(T, Constant(0.0), top)
BC_bottom_theta = DirichletBC(T, Constant(0.0), bottom)
BC_left_theta   = DirichletBC(T, Constant(0.0), left)
BC_right_theta  = DirichletBC(T, Constant(0.0), right)

# prescribed displacement boundary
presLoad    = Expression("t", t = 0.0, degree=1)
BC_top_pres1 = DirichletBC(U_0, presLoad, top)
BC_top_pres2 = DirichletBC(U_1, presLoad, top)

# displacement & micro-rotation boundary condition
BC = [BC_bottom,                      \
      BC_top_theta,  BC_bottom_theta, \
      BC_left_theta, BC_right_theta,  \
      BC_top_pres1,  BC_top_pres2]     
  
# phase-field boundary condition   
BC_d = []
  
# mark boundaries
boundaries = MeshFunction('size_t', mesh, dim-1)
boundaries.set_all(0)

top.mark(boundaries,1)
  
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

# Degradation function parameter
m = 3.*Gc/(8.*lc*psi_cr)


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


# Strain energy densities -------------------------
def psi_B(u):

  eps_sym = epsilon_sym(u)
  
  eps1 = (1./2.)*tr(eps_sym) + sqrt( (1./4.)*(tr(eps_sym)**2) - det(eps_sym) )
  eps2 = (1./2.)*tr(eps_sym) - sqrt( (1./4.)*(tr(eps_sym)**2) - det(eps_sym) )
  
  tr_eps_plus = (1./2.)*(tr(eps_sym) + abs(tr(eps_sym)))
  eps_plus_doubledot_eps_plus = ((1./2.)*(eps1 + abs(eps1)))**2 + ((1./2.)*(eps2 + abs(eps2)))**2

  energy_B = (1./2.)*lamda*(tr_eps_plus**2) + (mu+(kappa/2.))*eps_plus_doubledot_eps_plus
  
  return energy_B


def psi_C(u, theta):
  
  eps_skew = epsilon_skew(u, theta)
  
  eps_skew_doubledot_eps_skew = eps_skew[0,0]**2 + eps_skew[0,1]**2 + eps_skew[1,0]**2 + eps_skew[1,1]**2
  
  energy_C = (1./2.)*kappa*eps_skew_doubledot_eps_skew
  
  return energy_C


def psi_R(theta):
  
  curvature = phi(theta)
  
  curvature_dot_curvature = curvature[0]**2 + curvature[1]**2

  energy_R = (1./2.)*gamma*curvature_dot_curvature
  
  return energy_R


def psi(u, theta):
  
  if degradation == 'B':
    energy = psi_B(u)
  elif degradation == 'C':
    energy = psi_C(u, theta)
  elif degradation == 'R':
    energy = psi_R(theta)
  elif degradation == 'B+C':
    energy = psi_B(u) + psi_C(u, theta)
  elif degradation == 'B+R':
    energy = psi_B(u) + psi_R(theta)
  elif degradation == 'B+C+R':
    energy = psi_B(u) + psi_C(u, theta) + psi_R(theta)
  
  return energy
# -------------------------------------------------
  

# Driving force -----------------------------------
def H(u_old, theta_old, u_new, theta_new, H_old):
  
  psi_i_new = psi(u_new,theta_new) - psi_cr
  psi_i_old = psi(u_old,theta_old) - psi_cr
  
  psi_new =  psi_cr + (1./2.)*(psi_i_new + abs(psi_i_new))
  psi_old =  psi_cr + (1./2.)*(psi_i_old + abs(psi_i_old))

  return conditional(lt(psi_old, psi_new), psi_new, H_old)
# -------------------------------------------------


# Degradation function & its derivative -----------
def g_d(d):

  numerator   = (1.-d)**2
  denominator = (1.-d)**2 + m*d*(1.+p*d)

  g_d_val = numerator/denominator
  
  return g_d_val


def g_d_prime(d):

  numerator   = (d-1.)*(d*(2.*p+1.) + 1.)*m
  denominator = ((d**2)*(m*p+1.) + d*(m-2.) + 1.)**2

  g_d_prime_val = numerator/denominator
  
  return g_d_prime_val
# -------------------------------------------------




  
# ---------------------------------------------------------------- 
# Define variational form
# ----------------------------------------------------------------
# Define test & trial spaces 
eta, xi = TestFunctions(V)
zeta    = TestFunction(W)

x_new = Function(V)
u_new, theta_new = split(x_new)

x_old = Function(V)
u_old, theta_old = split(x_old)

d_new = Function(W)
d_old = Function(W) 

H_old = Function(W)

del_x = TrialFunction(V)
del_d = TrialFunction(W)

# Weak form: balance equations
if degradation == 'B':
  G_ut = g_d(d_new) * inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       +              inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       +              inner(phi(xi), m_R(theta_new)) * dx                     \
       -              inner(xi, E3_sigma_C(u_new, theta_new)) * dx
    
elif degradation == 'C':
  G_ut =              inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       + g_d(d_new) * inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       +              inner(phi(xi), m_R(theta_new)) * dx                     \
       - g_d(d_new) * inner(xi, E3_sigma_C(u_new, theta_new)) * dx
        
elif degradation == 'R':
  G_ut =              inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       +              inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       + g_d(d_new) * inner(phi(xi), m_R(theta_new)) * dx                     \
       -              inner(xi, E3_sigma_C(u_new, theta_new)) * dx
        
elif degradation == 'B+C':
  G_ut = g_d(d_new) * inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       + g_d(d_new) * inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       +              inner(phi(xi), m_R(theta_new)) * dx                     \
       - g_d(d_new) * inner(xi, E3_sigma_C(u_new, theta_new)) * dx
      
elif degradation == 'B+R':
  G_ut = g_d(d_new) * inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       +              inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       + g_d(d_new) * inner(phi(xi), m_R(theta_new)) * dx                     \
       -              inner(xi, E3_sigma_C(u_new, theta_new)) * dx
    
elif degradation == 'B+C+R':
  G_ut = g_d(d_new) * inner(epsilon(eta, xi), sigma_B(u_new)) * dx            \
       + g_d(d_new) * inner(epsilon(eta, xi), sigma_C(u_new, theta_new)) * dx \
       + g_d(d_new) * inner(phi(xi), m_R(theta_new)) * dx                     \
       - g_d(d_new) * inner(xi, E3_sigma_C(u_new, theta_new)) * dx


J_ut = derivative(G_ut, x_new, del_x) # jacobian

# Weak form: phase-field equation
G_d = H(u_old, theta_old, u_new, theta_new, H_old)*inner(zeta, g_d_prime(d_new)) * dx \
    + (3.*Gc/(8.*lc)) * (zeta + (2.*lc**2)*inner(grad(zeta), grad(d_new))) * dx  

J_d = derivative(G_d, d_new, del_d) # jacobian

# Constraints for the phase field
d_min = interpolate(Constant(DOLFIN_EPS), W) # lower bound
d_max = interpolate(Constant(1.0), W)        # upper bound

# Problem definition
p_ut = NonlinearVariationalProblem(G_ut, x_new, BC,   J_ut)
p_d  = NonlinearVariationalProblem(G_d,  d_new, BC_d, J_d)
p_d.set_bounds(d_min, d_max) # set bounds for the phase field

# Construct solvers
solver_ut = NonlinearVariationalSolver(p_ut)
solver_d  = NonlinearVariationalSolver(p_d)

# Set nonlinear solver parameters
newton_prm = solver_ut.parameters['newton_solver']
newton_prm['relative_tolerance'] = newton_Rtol
newton_prm['absolute_tolerance'] = newton_Atol
newton_prm['maximum_iterations'] = newton_maxiter
newton_prm['error_on_nonconvergence'] = False

snes_prm = {"nonlinear_solver": "snes",
            "snes_solver"     : { "method": "vinewtonssls",
                                  "line_search": "basic",
                                  "maximum_iterations": snes_maxiter,
                                  "relative_tolerance": snes_Rtol,
                                  "absolute_tolerance": snes_Atol,
                                  "report": True,
                                  "error_on_nonconvergence": False,
                                }}
solver_d.parameters.update(snes_prm)





# ---------------------------------------------------------------- 
# Solve system & output results
# ----------------------------------------------------------------
vtkfile_u     = File('./'+file_name+'/'+file_name+'_u.pvd')
vtkfile_theta = File('./'+file_name+'/'+file_name+'_theta.pvd')
vtkfile_d     = File('./'+file_name+'/'+file_name+'_d.pvd')

t  = t_i
while t <= t_f:
  
  t += dt
  
  print(' ')
  print('=================================================================================')
  print('>> t =', t, '[sec]')
  print('=================================================================================')

  presLoad.t = t*disp_rate
  
  iter = 0
  err  = 1
  
  while err > staggered_tol:
    iter += 1

    print('---------------------------------------------------------------------------------')
    print('>> iter.', iter)
    print('---------------------------------------------------------------------------------')
    
    # solve phase field equation
    print('[Solving phase field equation...]')
    solver_d.solve()
    
    # solve momentum balance equations
    print(' ')
    print('[Solving balance equations...]')
    solver_ut.solve()
    
    # compute error norms
    print(' ')
    print('[Computing error norms...]')   
    u_new, theta_new = x_new.split()
    u_old, theta_old = x_old.split()

    err_u     = errornorm(u_new, u_old,         norm_type = 'l2', mesh = None)
    err_theta = errornorm(theta_new, theta_old, norm_type = 'l2', mesh = None)
    err_d     = errornorm(d_new, d_old,         norm_type = 'l2', mesh = None)
    err = max(err_u, err_theta, err_d)

    x_old.assign(x_new)
    d_old.assign(d_new)
    H_old.assign(project(\
      conditional( \
        lt(H_old, psi_cr + (1./2.)*(psi(u_new,theta_new)-psi_cr + abs(psi(u_new,theta_new)-psi_cr))), \
          psi_cr + (1./2.)*(psi(u_new,theta_new)-psi_cr + abs(psi(u_new,theta_new)-psi_cr)),          \
          H_old                                                                                       \
      ), WW))

    if err < staggered_tol or iter >= staggered_maxiter:
    
      print('=================================================================================')
      print(' ')

      u_new.rename("Displacement", "label")
      theta_new.rename("Micro-rotation", "label")
      d_new.rename("Phase field", "label")
       
      vtkfile_u     << u_new
      vtkfile_theta << theta_new
      vtkfile_d     << d_new

      break

toc = time.time() - tic

print('Elapsed CPU time: ', toc, '[sec]')