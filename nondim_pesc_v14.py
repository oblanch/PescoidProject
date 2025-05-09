import numpy as np
import os
from dolfin import *
from tqdm import tqdm
from ufl import tanh

def run_non_dim_sim(length_scale, Delta, Flow, tau_m, Gamma, Activity, Beta, Sigma_c, R, delta_t, feedback):
    set_log_level(LogLevel.ERROR)
    # Parameters
    t_g = 30
    Total_time = 60 * 12  # minutes
    T = Total_time / t_g  # final time (nondim)
    dt = delta_t  # time step size
    num_steps = int(round(T / dt))  # number of time steps
    save_interval = 10

    L_0 = length_scale
    delta = Delta
    F = Flow
    T_m = tau_m
    gamma = Gamma
    A = Activity
    sigma_c = Sigma_c
    beta = Beta
    r = R
    rho_sens = 0.1
    param_list = [length_scale, Delta, Flow, tau_m, Gamma, Activity, Beta, Sigma_c, R]

    h_0 = 1.0
    sens = 1e-2
    c = 0.2  # Leading edge threshold density
    x_interval = 5e-3
    domain_length = 10
    c_0 = domain_length / 2
    nx = int(round(domain_length / x_interval))

    # Create mesh and define function space
    mesh = IntervalMesh(nx, -c_0, c_0)
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)

    x_coords = mesh.coordinates().flatten()  # Record x-coordinates for plotting
    y = np.round(c_0/x_interval)
    y=int(y)
    print(f'time step: {dt}')
    print(f'x-interval: {x_interval}')

    # Initial conditions
    rho_0_expr = '(h_0/2) * ((-tanh((pow(x[0], 2)-L_0)/sens)) + 1)*pow(0.5,x[0]*x[0])'
    m_0_expr = '-1'
    u_0_expr = '0.0'

    # Create expressions for initial conditions
    rho_0 = Expression(rho_0_expr, degree=2, L_0=L_0, sens=sens, h_0=h_0)
    m_0 = Expression(m_0_expr, degree=2, L_0=L_0, sens=sens, h_0=h_0)
    u_0 = Expression(u_0_expr, degree=1)

    # Initialize solution function
    k_n = Function(V)
    assign(k_n.sub(0), interpolate(rho_0, V.sub(0).collapse()))
    assign(k_n.sub(1), interpolate(m_0, V.sub(1).collapse()))
    assign(k_n.sub(2), interpolate(u_0, V.sub(2).collapse()))

    # Extract initial conditions
    #rho_n, m_n, u_n, s_n = split(k_n)
    rho_n, m_n, u_n = split(k_n)

    # Define trial and test functions
    #rho, m, u, s = TrialFunctions(V)
    rho, m, u = TrialFunctions(V)

    #phi, psi, eta, zeta = TestFunctions(V)
    phi, psi, eta = TestFunctions(V)

    # Define source terms and variational problem
    F_rho = (
        (rho - rho_n) * phi * dx
        + dt * delta * rho.dx(0) * phi.dx(0) * dx
        - dt * F * u_n * rho_n * phi.dx(0) * dx
        - dt * rho_n * (1-rho_n) * phi * dx
    )

    m_sens = 1e-1

    if feedback == 'strain rate':
        F_m = (
            (m - m_n) * psi * dx
            - dt * (1/T_m) * m_n * (m_n+1) * (1-m_n) * psi * dx
            #- dt * r * (1/2)*(tanh((rho_n-0.2)/0.1)+1) * ((rho_n * (A * rho_n * (1 + beta*((tanh(m_n/m_sens)+1)/2)) - 1) + u_n.dx(0))-sigma_c) * psi * dx
            - dt * (1/T_m) * r * ((rho_n * (A * (rho_n/(1+rho_sens*(rho_n*rho_n))) * (1 + beta*((tanh((m_n-m_sens)/m_sens)+1)/2))))-sigma_c) * psi * dx
            + dt * delta * m.dx(0) * psi.dx(0) * dx
            + dt * F  * u_n * m_n.dx(0) * psi * dx
        )
    
    if feedback == 'active stress':
        F_m = (
            (m - m_n) * psi * dx
            - dt * (1/T_m) * m_n * (m_n+1) * (1-m_n) * psi * dx
            - dt * (1/T_m) * r * (sigma_c - u_n.dx(0)) * psi * dx
            + dt * delta * m.dx(0) * psi.dx(0) * dx
            + dt * F  * u_n * m_n.dx(0) * psi * dx
        )

    F_u = (
        (1/2)*(tanh((rho_n-0.02)/0.01)+1) * gamma * u * eta * dx
        + (1/2)*(tanh((rho_n-0.02)/0.01)+1) * u.dx(0) * eta.dx(0) * dx
        - (rho_n * (A * (rho_n/(1+rho_sens*(rho_n*rho_n))) * (1 + beta*((tanh((m_n-m_sens)/m_sens)+1)/2)) - 1)).dx(0) * eta * dx
    )





    F = F_rho + F_m + F_u# + F_s
    a, L = lhs(F), rhs(F)

    # Solving the coupled system
    k = Function(V)
    t = 0
    output_dir = 'simulation_results'
    os.makedirs(output_dir, exist_ok=True)


    aborted = False

    boundary_positions = []
    boundary_time_data = []
    meso_frac_data = []
    boundary_velocity_data = []

    density_data = []
    mesoderm_data = []
    max_mesoderm_data = []
    velocity_data = []
    time_data = []
    stress_data = []

    # Finding the largest x-coordinate where rho is close to the threshold c
    def find_largest_x(rho, mesh, c):
        x_coords = mesh.coordinates().flatten()
        rho_values = rho.compute_vertex_values(mesh)
        max_x = -np.inf
        max_x_index = None
        for i in range(len(rho_values)):
            if np.abs(rho_values[i] - c) < 1e-1 and x_coords[i] > 0:
                max_x = max(max_x, x_coords[i])
                max_x_index = i
        return max_x if max_x != -np.inf else None, max_x_index
    
    def find_largest_m(m,mesh):
        m_values = m.compute_vertex_values(mesh)
        max_m = max(m_values)
        return max_m
    
    def m_fraction(m,max_x_index,c_0,x_interval,mesh):
        m_value = m.compute_vertex_values(mesh)
        m_pos = []
        m_neg = []
        m_frac = []
        i_range = None

        start_index = int(np.round(c_0/x_interval))
        if max_x_index != None:
            i_range = range(start_index,max_x_index)
            for i in i_range:
                if m_value[i] > 0:
                    m_pos.append(m_value[i])  # Use append to add elements to the list
                else:
                    m_neg.append(m_value[i])
            m_frac = len(m_pos) / (len(m_neg)+len(m_pos))
        return m_frac
            
        
    
    def normalize_m(m):
        m_data = m
        m_min = np.min(m)
        m_max = np.max(m)
        m_data = [x - m_min for x in m_data]
        m_data = m_data/(m_max - m_min)
        return m_data


    # Ensuring that no rho values are negative
    def check_and_correct(variable):
        V = variable.function_space()
        corrected = Function(V)
        corrected_expr = conditional(lt(variable, 0), Constant(0), variable)
        corrected.assign(project(corrected_expr, V))
        return corrected
    
    def compute_stress(rho_n,m_n,A,beta,m_sens,rho_sens):
        # Create a scalar function space for the projection
        scalar_space = FunctionSpace(mesh, 'CG', 1)
        stress = project((rho_n * A * (rho_n/(1+rho_sens*(rho_n*rho_n))) * (1 + beta*((tanh((m_n-m_sens)/m_sens)+1)/2))), scalar_space)
        # Return the local array of values
        return stress.vector().get_local()
    
    def has_negative(arr):
        for x in len(arr):
            if x < 0:
                return True
            return False
        
    for n in range(num_steps):
        t += dt

        solve(a == L, k)

        rho_new, m_new, u_new  = k.split(deepcopy=True)
        
        if np.any(~np.isfinite(rho_new.vector().get_local())) or np.any(~np.isfinite(m_new.vector().get_local())):
            print("Aborting: NaN or Inf detected in rho_new or m_new.")
            aborted = True
            break

        rho_new = check_and_correct(rho_new)
        
        assign(k_n.sub(0), rho_new)
        assign(k_n.sub(1), m_new)
        assign(k_n.sub(2), u_new)

        k_n.assign(k)
        stress_new = compute_stress(rho_n,m_n,A,beta,m_sens,rho_sens)
        max_x = find_largest_x(rho_new, mesh, c)[0]
        max_x_index = find_largest_x(rho_new, mesh, c)[1]
        max_m = find_largest_m(m_new,mesh)
        m_frac = m_fraction(m_new,max_x_index,c_0,x_interval,mesh)
        
        if max_x is not None:
            boundary_positions.append(max_x)
            meso_frac_data.append(m_frac)
            boundary_time_data.append(t)
            max_x_index = np.where(np.isclose(x_coords, max_x, atol=1e-3))[0][0]
            boundary_velocity_data.append(u_new.vector().get_local()[max_x_index])
            max_mesoderm_data.append(max_m)


        if n % save_interval == 0:
            current_density = rho_new.vector().get_local()

            density_data.append(current_density.copy())
            mesoderm_data.append(m_new.vector().get_local().copy())
            velocity_data.append(u_new.vector().get_local().copy())
            stress_data.append(stress_new.copy())
            time_data.append(t)



    mesh_params = np.array([nx, domain_length])
    time_data = [x * t_g for x in time_data]
    boundary_time_data = [x * t_g for x in boundary_time_data]
    max_mesoderm_data = normalize_m(max_mesoderm_data)

    np.save(os.path.join(output_dir, 'density_data.npy'), density_data)
    np.save(os.path.join(output_dir, 'mesoderm_data.npy'), mesoderm_data)
    np.save(os.path.join(output_dir, 'max_mesoderm_data.npy'), max_mesoderm_data)
    np.save(os.path.join(output_dir, 'meso_frac_data.npy'), meso_frac_data)
    np.save(os.path.join(output_dir, 'stress_data.npy'), stress_data)
    np.save(os.path.join(output_dir, 'velocity_data.npy'), velocity_data)
    np.save(os.path.join(output_dir, 'time_data.npy'), time_data)
    np.save(os.path.join(output_dir, 'boundary_positions.npy'), boundary_positions)
    np.save(os.path.join(output_dir, 'boundary_time_data.npy'), boundary_time_data)
    np.save(os.path.join(output_dir, 'boundary_velocity_data.npy'), boundary_velocity_data)
    np.save(os.path.join(output_dir, 'mesh_params.npy'), mesh_params)
    np.save(os.path.join(output_dir, 'x_coords.npy'), x_coords)
    np.save(os.path.join(output_dir, 'params.npy'), param_list)

    return (boundary_positions, boundary_time_data, density_data, mesoderm_data, meso_frac_data, x_coords, aborted, dt)