from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from fancy_plots import fancy_plots_2, fancy_plots_1
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

# Global variables Odometry Drone
x_real = 0.0
y_real = 0.0
z_real = 0.0
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0

# Angular velocities
qx_real = 0.0005
qy_real = 0.0
qz_real = 0.0
qw_real = 1.0
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0


def odometry_call_back(odom_msg):
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    # Read desired linear velocities from node
    x_real = odom_msg.pose.pose.position.x 
    y_real = odom_msg.pose.pose.position.y
    z_real = odom_msg.pose.pose.position.z
    vx_real = odom_msg.twist.twist.linear.x
    vy_real = odom_msg.twist.twist.linear.y
    vz_real = odom_msg.twist.twist.linear.z


    qx_real = odom_msg.pose.pose.orientation.x
    qy_real = odom_msg.pose.pose.orientation.y
    qz_real = odom_msg.pose.pose.orientation.z
    qw_real = odom_msg.pose.pose.orientation.w

    wx_real = odom_msg.twist.twist.angular.x
    wy_real = odom_msg.twist.twist.angular.y
    wz_real = odom_msg.twist.twist.angular.z
    return None


def Rot_zyx(x):
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Rot Matrix axis X
    RotX = MX.zeros(3, 3)
    RotX[0, 0] = 1.0
    RotX[1, 1] = cos(phi)
    RotX[1, 2] = -sin(phi)
    RotX[2, 1] = sin(phi)
    RotX[2, 2] = cos(phi)

    # Rot Matrix axis Y
    RotY = MX.zeros(3, 3)
    RotY[0, 0] = cos(theta)
    RotY[0, 2] = sin(theta)
    RotY[1, 1] = 1.0
    RotY[2, 0] = -sin(theta)
    RotY[2, 2] = cos(theta)

    RotZ = MX.zeros(3, 3)
    RotZ[0, 0] = cos(psi)
    RotZ[0, 1] = -sin(psi)
    RotZ[1, 0] = sin(psi)
    RotZ[1, 1] = cos(psi)
    RotZ[2, 2] = 1.0

    R = RotZ@RotY@RotX
    return R
def M_matrix_bar(chi, x):

    # Split Parameters
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Constants of the system
    m1 = chi[0];
    Ixx = chi[1];
    Iyy = chi[2];
    Izz = chi[3];

    # Mass Matrix
    M = MX.zeros(6, 6)
    M[0, 0] = m1
    M[1, 1] = m1
    M[2, 2] = m1
    M[3, 3] = Ixx
    M[3, 5] = -Ixx*sin(theta)
    M[4, 4] = Izz + Iyy*cos(phi)**2 - Izz*cos(phi)**2
    M[4, 5] = Iyy*cos(phi)*cos(theta)*sin(phi) - Izz*cos(phi)*cos(theta)*sin(phi)
    M[5, 3] = -Ixx*sin(theta)
    M[5, 4] = Iyy*cos(phi)*cos(theta)*sin(phi) - Izz*cos(phi)*cos(theta)*sin(phi)
    M[5, 5] = Ixx - Ixx*cos(theta)**2 + Iyy*cos(theta)**2 - Iyy*cos(phi)**2*cos(theta)**2 + Izz*cos(phi)**2*cos(theta)**2
    return M

def C_matrix_bar(chi, x):
    # Split Parameters system
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    phi_p = x[9, 0]
    theta_p = x[10, 0]
    psi_p = x[11, 0]

    m1 = chi[0];
    Ixx = chi[1];
    Iyy = chi[2];
    Izz = chi[3];

    C = MX.zeros(6, 6)
    C[3, 4] = (Iyy*psi_p*cos(theta))/2 - (Ixx*psi_p*cos(theta))/2 - (Izz*psi_p*cos(theta))/2 - Iyy*psi_p*cos(phi)**2*cos(theta) + Izz*psi_p*cos(phi)**2*cos(theta) + Iyy*theta_p*cos(phi)*sin(phi) - Izz*theta_p*cos(phi)*sin(phi)
    C[3, 5] = (Iyy*theta_p*cos(theta))/2 - (Ixx*theta_p*cos(theta))/2 - (Izz*theta_p*cos(theta))/2 - Iyy*theta_p*cos(phi)**2*cos(theta) + Izz*theta_p*cos(phi)**2*cos(theta) - Iyy*psi_p*cos(phi)*cos(theta)**2*sin(phi) + Izz*psi_p*cos(phi)*cos(theta)**2*sin(phi)
    C[4, 3] = (Ixx*psi_p*cos(theta))/2 - (Iyy*psi_p*cos(theta))/2 + (Izz*psi_p*cos(theta))/2 + Iyy*psi_p*cos(phi)**2*cos(theta) - Izz*psi_p*cos(phi)**2*cos(theta) - Iyy*theta_p*cos(phi)*sin(phi) + Izz*theta_p*cos(phi)*sin(phi)
    C[4, 4] = Izz*phi_p*cos(phi)*sin(phi) - Iyy*phi_p*cos(phi)*sin(phi)
    C[4, 5] = (Ixx*phi_p*cos(theta))/2 - (Iyy*phi_p*cos(theta))/2 + (Izz*phi_p*cos(theta))/2 + Iyy*phi_p*cos(phi)**2*cos(theta) - Izz*phi_p*cos(phi)**2*cos(theta) - Ixx*psi_p*cos(theta)*sin(theta) + Iyy*psi_p*cos(theta)*sin(theta) - Iyy*psi_p*cos(phi)**2*cos(theta)*sin(theta) + Izz*psi_p*cos(phi)**2*cos(theta)*sin(theta)
    C[5, 3] = (Izz*theta_p*cos(theta))/2 - (Iyy*theta_p*cos(theta))/2 - (Ixx*theta_p*cos(theta))/2 + Iyy*theta_p*cos(phi)**2*cos(theta) - Izz*theta_p*cos(phi)**2*cos(theta) + Iyy*psi_p*cos(phi)*cos(theta)**2*sin(phi) - Izz*psi_p*cos(phi)*cos(theta)**2*sin(phi)
    C[5, 4] = (Izz*phi_p*cos(theta))/2 - (Iyy*phi_p*cos(theta))/2 - (Ixx*phi_p*cos(theta))/2 + Iyy*phi_p*cos(phi)**2*cos(theta) - Izz*phi_p*cos(phi)**2*cos(theta) + Ixx*psi_p*cos(theta)*sin(theta) - Iyy*psi_p*cos(theta)*sin(theta) + Iyy*psi_p*cos(phi)**2*cos(theta)*sin(theta) - Izz*psi_p*cos(phi)**2*cos(theta)*sin(theta) - Iyy*theta_p*cos(phi)*sin(phi)*sin(theta) + Izz*theta_p*cos(phi)*sin(phi)*sin(theta)
    C[5, 5] = Ixx*theta_p*cos(theta)*sin(theta) - Iyy*theta_p*cos(theta)*sin(theta) + Iyy*phi_p*cos(phi)*cos(theta)**2*sin(phi) - Izz*phi_p*cos(phi)*cos(theta)**2*sin(phi) + Iyy*theta_p*cos(phi)**2*cos(theta)*sin(theta) - Izz*theta_p*cos(phi)**2*cos(theta)*sin(theta)
    return C

def G_matrix_bar(chi, x):
    g = 9.81

    # Split Parameters of the system
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Constan values of the system
    m1 = chi[0];
    G = MX.zeros(6, 1)
    G[2, 0] = g*m1
    return G

def S_fuction(chi):
    S = MX.zeros(6, 6)
    S[2, 2] = chi[4]
    S[3, 3] = chi[5]
    S[4, 4] = chi[6]
    S[5, 5] = chi[7]
    return S
def Q_fuction(chi):
    Q = MX.zeros(6, 6)
    Q[3, 3] = chi[8]
    Q[4, 4] = chi[9]
    Q[5, 5] = chi[10]
    return Q

def E_fuction(chi):
    E = MX.zeros(6, 6)
    E[2,2] = chi[11];
    E[3,3] = chi[12];
    E[4,4] = chi[13];
    E[5,5] = chi[14];
    return E

def T_fuction(chi):
    E = MX.zeros(6,6)
    E[2,2] = chi[15]
    return E
def B_fuction(chi):
    m1 = chi[0];
    g = 9.81
    B = MX.zeros(6,1)
    B[2, 0] = m1*g
    return B

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system
    g = 9.81
    chi = [1.05833614147124e-08, 1.71991420525648e-08, 1.18746744226948e-08, 5.75996411364598e-08, 6.16080761085103e-08, 1.49712137729156e-06, 8.42430756632765e-07, 1.34018154831176e-06, 1.33788423506417e-06, 7.32461559124120e-07, 1.30714877460439e-06, 6.07978755974178e-08, 3.08456881692582e-07, 1.79780921983864e-07, 4.25094296520110e-07, 6.08414838608068e-08]
    m = chi[0]

    # set up states & controls
    # Position
    x1 = MX.sym('x1')
    y1 = MX.sym('y1')
    z1 = MX.sym('z1')
    # Orientation
    phi = MX.sym('phi')
    theta = MX.sym('theta')
    psi = MX.sym('psi')

    # Velocity Linear and Angular
    dx1 = MX.sym('dx1')
    dy1 = MX.sym('dy1')
    dz1 = MX.sym('dz1')
    dphi = MX.sym('dphi')
    dtheta = MX.sym('dtheta')
    dpsi = MX.sym('dpsi')

    # General vector of the states
    x = vertcat(x1, y1, z1, phi, theta, psi, dx1, dy1, dz1, dphi, dtheta, dpsi)

    # Action variables
    zp_ref = MX.sym('F')
    phi_ref = MX.sym('ux')
    theta_ref = MX.sym('uy')
    psi_ref = MX.sym('uz')

    # General Vector Action variables
    u = vertcat(zp_ref, phi_ref, theta_ref, psi_ref)

    # Variables to explicit function
    x1_dot = MX.sym('x1_dot')
    y1_dot = MX.sym('y1_dot')
    z1_dot = MX.sym('z1_dot')
    phi_dot = MX.sym('phi_dot')
    theta_dot = MX.sym('theta_dot')
    psi_dot = MX.sym('psi_dot')
    dx1_dot = MX.sym('dx1_dot')
    dy1_dot = MX.sym('dy1_dot')
    dz1_dot = MX.sym('dz1_dot')
    dphi_dot = MX.sym('dphi_dot')
    dtheta_dot = MX.sym('dtheta_dot')
    dpsi_dot = MX.sym('dpsi_dot')

    # general vector X dot for implicit function
    xdot = vertcat(x1_dot, y1_dot, z1_dot,  phi_dot, theta_dot, psi_dot, dx1_dot, dy1_dot, dz1_dot, dphi_dot, dtheta_dot, dpsi_dot)

    # Rotational Matrix
    R = Rot_zyx(x);
    M_bar = M_matrix_bar(chi, x)
    C_bar = C_matrix_bar(chi, x)
    G_bar = G_matrix_bar(chi, x)
    S = S_fuction(chi)
    Q = Q_fuction(chi)
    E = E_fuction(chi)
    T = T_fuction(chi)
    B = B_fuction(chi)

    # Auxiliar Matrices 
    R_t = MX.zeros(6, 6)
    R_t[0:3, 0:3] = R@T[0:3, 0:3]
    R_t[0:3, 3:6] = T[0:3, 3:6]
    R_t[3:6, 0:3] = T[3:6, 0:3]
    R_t[3:6, 3:6] = T[3:6, 3:6]

    # Aux Control
    u_aux = vertcat(0, 0, zp_ref, phi_ref, theta_ref, psi_ref)

    Aux = S@u_aux-Q@x[0:6, 0]-E@x[6:12, 0]+B

    Aux1 = R@Aux[0:3,0]
    Aux2 = Aux[3:6,0]

    # New Input Model
    Input_model = MX.zeros(6, 1)
    Input_model[0:3,0] = Aux1
    Input_model[3:6,0] = Aux2

    # Aux inverse Matrix
    M_a_r = M_bar + R_t
    inv_M = inv(M_a_r)

    x_pp = inv_M@(Input_model-C_bar@x[6:12, 0]-G_bar);

    f_expl = MX.zeros(12, 1)
    f_expl[0:6, 0] = x[6:12, 0]
    f_expl[6:12, 0] = x_pp

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model, f_system

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((12,))
    return aux_x

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_model()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = 1 * np.diag([2, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x,th,dx,dth]
    R_mat = 1 * np.diag([(1/zp_max),  (1/phi_max), (1/theta_max), (1/psi_max)])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([zp_min, phi_min, theta_min, psi_min])
    ocp.constraints.ubu = np.array([zp_max, phi_max, theta_max, psi_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp
def get_odometry(pose, velocity):
    displacement = [pose[0,0], pose[0,1], pose[0,2]]
    quaternion = [pose[0,3], pose[0,4], pose[0,5], pose[0,6]]
    r_quat = R.from_quat(quaternion)
    euler =  r_quat.as_euler('xyz', degrees = False)
    linear_velocity = [velocity[0,0], velocity[0,1], velocity[0,2]]
    angular_velocity = [velocity[0,3], velocity[0,4], velocity[0,5]]
    state = np.array([displacement[0], displacement[1], displacement[2], euler[0], euler[1], euler[2], linear_velocity[0], linear_velocity[1], linear_velocity[2], angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    print(state)
    return None

def main():
    # Initial Values System
    # Simulation Time
    t_final = 30
    # Sample time
    t_s = 0.03
    # Prediction Time
    t_prediction= 2;

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)


    # Vector Initial conditions
    x = np.zeros((12, t.shape[0]+1-N_prediction), dtype = np.double)
    x[0,0] = 0.0
    x[1,0] = 0.0
    x[2,0] = 0.0
    x[3,0] = 0*(np.pi)/180
    x[4,0] = 0*(np.pi)/180
    x[5,0] = 0*(np.pi)/180
    x[6,0] = 0.0
    x[7,0] = 0.0
    x[8,0] = 0.0
    x[9,0] = 0.0
    x[10,0] = 0.0
    x[11,0] = 0.0

    # Read Values Odometry Drone
    data_pose = np.array([[x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real]], dtype=np.double)
    data_velocity = np.array([[vx_real, vy_real, vz_real, wx_real, wy_real, wz_real]], dtype=np.double)

    # Reference Signal of the system
    xref = np.zeros((16, t.shape[0]), dtype = np.double)
    xref[0,:] = 4 * np.sin(1*t)+3;
    xref[1,:] =  4 * np.sin(1.5*t);
    xref[2,:] = 2.5 * np.sin (1* t) +5 
    xref[5,:] = 0*np.pi/180
    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 3
    phi_max = 0.5
    theta_max = 0.5
    psi_max = 0.5

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_min = -psi_max

    # Create Optimal problem
    model, f = f_system_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    #AcadosOcpSolver.generate(ocp, json_file=solver_json)
    #AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    #acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    # Simulation System

    for k in range(0, t.shape[0]-N_prediction):
        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "yref", yref)
        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "yref", yref_N[0:12])

        # Get Computational Time
        data_pose = np.array([[x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real]], dtype=np.double)
        data_velocity = np.array([[vx_real, vy_real, vz_real, wx_real, wy_real, wz_real]], dtype=np.double)
        get_odometry(data_pose, data_velocity)
        tic = time.time()
        status = acados_ocp_solver.solve()

        toc = time.time()- tic
        print(toc)

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        # System Evolution
        x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
        delta_t[:, k] = toc

    fig1, ax11 = fancy_plots_1()
    states_x, = ax11.plot(t[0:x.shape[1]], x[0,:],
                    color='#BB5651', lw=2, ls="-")
    states_y, = ax11.plot(t[0:x.shape[1]], x[1,:],
                    color='#69BB51', lw=2, ls="-")
    states_z, = ax11.plot(t[0:x.shape[1]], x[2,:],
                    color='#5189BB', lw=2, ls="-")
    states_xd, = ax11.plot(t[0:x.shape[1]], xref[0,0:x.shape[1]],
                    color='#BB5651', lw=2, ls="--")
    states_yd, = ax11.plot(t[0:x.shape[1]], xref[1,0:x.shape[1]],
                    color='#69BB51', lw=2, ls="--")
    states_zd, = ax11.plot(t[0:x.shape[1]], xref[2,0:x.shape[1]],
                    color='#5189BB', lw=2, ls="--")

    ax11.set_ylabel(r"$[states]$", rotation='vertical')
    ax11.set_xlabel(r"$[t]$", labelpad=5)
    ax11.legend([states_x, states_y, states_z, states_xd, states_yd, states_zd],
            [r'$x$', r'$y$', r'$z$', r'$x_d$', r'$y_d$', r'$z_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig1.savefig("states_xyz.eps")
    fig1.savefig("states_xyz.png")
    fig1
    plt.show()

    fig2, ax12 = fancy_plots_1()
    states_phi, = ax12.plot(t[0:x.shape[1]], x[3,:],
                    color='#BB5651', lw=2, ls="-")
    states_theta, = ax12.plot(t[0:x.shape[1]], x[4,:],
                    color='#69BB51', lw=2, ls="-")
    states_psi, = ax12.plot(t[0:x.shape[1]], x[5,:],
                    color='#5189BB', lw=2, ls="-")

    ax12.set_ylabel(r"$[states]$", rotation='vertical')
    ax12.set_xlabel(r"$[t]$", labelpad=5)
    ax12.legend([states_phi, states_theta, states_psi],
            [r'$\phi$', r'$\theta$', r'$\psi$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig2.savefig("states_angles.eps")
    fig2.savefig("states_angles.png")
    fig2
    plt.show()

    fig3, ax13 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax13.set_xlim((t[0], t[-1]))

    time_1, = ax13.plot(t[0:delta_t.shape[1]],delta_t[0,:],
                    color='#00429d', lw=2, ls="-")
    tsam1, = ax13.plot(t[0:t_sample.shape[1]],t_sample[0,:],
                    color='#9e4941', lw=2, ls="-.")

    ax13.set_ylabel(r"$[s]$", rotation='vertical')
    ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax13.legend([time_1,tsam1],
            [r'$t_{compute}$',r'$t_{sample}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig3.savefig("time.eps")
    fig3.savefig("time.png")
    fig3
    plt.show()

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        odometry_topic = "/drone/odometry"
        velocity_subscriber = rospy.Subscriber(odometry_topic, Odometry, odometry_call_back)
        main()
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass
