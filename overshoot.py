###### COMMAND TO EXECUTE BELOW ######
# BE SURE YOU ARE IN MAIN DIRECTORY (Kwolek_Andrew_capstone/)
# python3 code/overshoot.py

import numpy as np
from modern_robotics import CartesianTrajectory, TransInv, MatrixLog6, se3ToVec, Adjoint, FKinBody, JacobianBody, NearZero, AxisAng6, TransToRp
import matplotlib.pyplot as plt
import logging

# Define parameters
joint_limits = [(-1.0, 1.0), (-1.117, 1.5), (-1.6, -0.2), (-1.5, -0.15), None]

block_i = np.array([1, 0, 0])
block_f = np.array([0, -1, -np.pi/2])

dt = 0.01
l = 0.235
w = 0.15
r = 0.0475

F = (r/4)*np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                    [1,      1,       1,        1],
                    [-1,      1,      -1,        1]])

OPEN = 0
CLOSE = 1

T_sb_0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0.0963],
                   [0, 0, 0, 1]])

T_b0 = np.array([[1, 0, 0, 0.1662],
                 [0, 1, 0,      0],
                 [0, 0, 1, 0.0026],
                 [0, 0, 0,      1]])

# Home configuration of arm
M_0e = np.array([[1, 0, 0,  0.033],
                 [0, 1, 0,      0],
                 [0, 0, 1, 0.6546],
                 [0, 0, 0,      1]])

# Screw axes in home configuration
B_list = np.array([[0,       0,       0,       0, 0],
                   [0,      -1,      -1,      -1, 0],
                   [1,       0,       0,       0, 1],
                   [0, -0.5076, -0.3526, -0.2176, 0],
                   [0.033,   0,       0,       0, 0],
                   [0,       0,       0,       0, 0]])

d1_max = 0.07
d1_min = 0.02
d2 = 0.035
d3 = 0.043

so = 0.2

cube_dim = np.array([0.05, 0.05, 0.05])
cube_pos_i = np.array([1, 0, 0.025])
cube_pos_f = np.array([0, -1, 0.025])

T_ce_grasp = np.array([[-np.cos(np.pi/6), 0, np.sin(np.pi/6), d3/4],
                       [0, 1, 0, 0],
                       [-np.sin(np.pi/6), 0, -np.cos(np.pi/6), 0],
                       [0, 0, 0, 1]])

T_ce_standoff = np.array([[-np.cos(np.pi/6), 0, np.sin(np.pi/6), d3/4],
                          [0, 1, 0, 0],
                          [-np.sin(np.pi/6), 0, -np.cos(np.pi/6), so],
                          [0, 0, 0, 1]])


# Milestone 1
def NextState(thetalist, thetadotlist, dt, max_vel) -> list:
    """Computes the next state of the system

    :param thetalist: List of configuration variables
    :param thetadotlist: List of configuration velocities
    :param dt: Time step
    :param max_vel: Maximum joint velocity

    :return: Configuration of the next state

    """
    for i in range(len(thetadotlist)):
        if thetadotlist[i] > max_vel:
            thetadotlist[i] = max_vel
        elif thetadotlist[i] < -max_vel:
            thetadotlist[i] = -max_vel
    chassis_list = np.array(thetalist[0:3])
    arm_joint_angles = np.array(thetalist[3:8])
    wheel_angles = np.array(thetalist[8:12])
    arm_joint_speeds = np.array(thetadotlist[4:])
    wheel_speeds = np.array(thetadotlist[0:4])

    new_arm_joint_angles = arm_joint_angles + arm_joint_speeds * dt
    du = wheel_speeds * dt
    new_wheel_angles = wheel_angles + du

    Vb = F @ du

    new_chassis_list = chassis_list + Vb

    new_state = new_chassis_list.tolist() + new_arm_joint_angles.tolist() + \
        new_wheel_angles.tolist()
    return new_state


# Milestone 2
def TrajectoryGenerator(T_se_init, T_sc_init, T_sc_final, T_ce_grasp, T_ce_standoff, k=1):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
      the screw motion about a space screw axis

    :param T_se_init: The initial end-effector configuration
    :param T_sc_init: The cube's initial configuration
    :param T_sc_final: The cube's final configuration
    :param T_ce_grasp: The end effector's configuration relative to
                       the cube when it is grasping the cube
    :param T_ce_standoff: The end effector's standoff configuration 
                          above the cube, before and after grasping,
                          relative to the cube
    :param k: The number of trajectory reference configurations per
              0.01 seconds
    :return: A concatenated list of all of the trajectory segments.

    """
    traj = []
    T_se_standoff_i = T_sc_init @ T_ce_standoff
    T = TransInv(T_se_init) @ T_se_standoff_i
    V_e = se3ToVec(MatrixLog6(T))
    theta_max = AxisAng6(V_e)[1]
    _, d_max = TransToRp(T)
    d_max = np.linalg.norm(np.array(d_max))
    Tf_1 = max(d_max/0.5, theta_max/0.5)
    Tf_1 = np.round(Tf_1, 2)
    gripper = 0

    seg_1 = MatToVec(CartesianTrajectory(
        Xstart=T_se_init, Xend=T_se_standoff_i, Tf=Tf_1, N=(Tf_1*k)/dt, method=5), gripper)
    traj = traj + seg_1

    T_se_grasp_i = T_sc_init @ T_ce_grasp
    seg_2 = MatToVec(CartesianTrajectory(Xstart=T_se_standoff_i,
                     Xend=T_se_grasp_i, Tf=2, N=k/dt, method=5), gripper)
    traj = traj + seg_2

    gripper = CLOSE
    seg_3 = MatToVec(CartesianTrajectory(Xstart=T_se_grasp_i,
                     Xend=T_se_grasp_i, Tf=0.630, N=(0.630*k)/dt, method=3), gripper)
    traj = traj + seg_3

    seg_4 = MatToVec(CartesianTrajectory(Xstart=T_se_grasp_i,
                     Xend=T_se_standoff_i, Tf=2, N=(k)/dt, method=5), gripper)
    traj = traj + seg_4

    T_se_standoff_e = T_sc_final @ T_ce_standoff
    T = TransInv(T_se_init) @ T_se_standoff_e
    V_e = se3ToVec(MatrixLog6(T))
    theta_max = AxisAng6(V_e)[1]
    _, d_max = TransToRp(T)
    d_max = np.linalg.norm(np.array(d_max))
    Tf_4 = max(d_max/0.5, theta_max/0.5)
    Tf_4 = np.round(Tf_4, 2)

    seg_5 = MatToVec(CartesianTrajectory(
        Xstart=T_se_standoff_i, Xend=T_se_standoff_e, Tf=Tf_4, N=(Tf_4*k)/dt, method=5), gripper)
    traj = traj + seg_5

    T_se_grasp_e = T_sc_final @ T_ce_grasp
    seg_6 = MatToVec(CartesianTrajectory(Xstart=T_se_standoff_e,
                     Xend=T_se_grasp_e, Tf=2, N=(k)/dt, method=5), gripper)
    traj = traj + seg_6

    gripper = OPEN
    seg_7 = MatToVec(CartesianTrajectory(Xstart=T_se_grasp_e,
                     Xend=T_se_grasp_e, Tf=0.630, N=(0.630*k)/dt, method=3), gripper)
    traj = traj + seg_7

    seg_8 = MatToVec(CartesianTrajectory(Xstart=T_se_grasp_e,
                     Xend=T_se_standoff_e, Tf=2, N=(k)/dt, method=5), gripper)
    traj = traj + seg_8

    return traj


# Milestone 3
def FeedbackControl(X, X_d, X_d_next, Kp=np.zeros((4, 4)), Ki=np.zeros((4, 4)), dt=0.01):
    """Computes the control law of the current state

    :param X: Actual position of the end effector
    :param X_d: Desired position of the end effector
    :param X_d_next: Desired end effector configuration at
        next time step
    :param Kp: Proportional gain
    :param Ki: Integral gain
    :param dt: Time step

    :return: Commanded end effector twist
    :return: 6-vector of the error twist

    """

    X_inv = TransInv(X)
    Ad_XinvXd = Adjoint(X_inv @ X_d)
    X_err = MatrixLog6(X_inv @ X_d)
    error_integral = X_err * dt
    Vd = se3ToVec((1/dt)*MatrixLog6(TransInv(X_d) @ X_d_next))
    V = Ad_XinvXd @ Vd + se3ToVec(Kp @ X_err) + se3ToVec(Ki @ error_integral)
    return V, se3ToVec(X_err)


def JacobianEndeffector(thetalist):
    """Computes end effector Jacobian

    :param thetalist: youBot configuration

    :return: end effector jacobian

    """
    F6 = np.vstack((np.zeros((2, F.shape[1])), F, np.zeros((1, F.shape[1]))))
    T_0e = FKinBody(M_0e, B_list, thetalist[3:8])
    J_base = Adjoint(TransInv(T_b0 @ T_0e)) @ F6
    J_arm = JacobianBody(B_list, thetalist[3:8])
    J_e = np.hstack((J_base, J_arm))
    return J_e


def GetJointSpeeds(V, J_e, thetalist, dt):
    """Compute the joint speeds

    :param V: end effector twist
    :param J_e: end effector jacobian
    :param thetalist: youBot configuration
    :param dt: time step

    :return: list of configuration velocities

    """
    violated = True
    while violated:
        J_pinv = np.linalg.pinv(J_e, 0.001)
        thetadotlist = J_pinv @ V
        for i in range(len(thetadotlist)):
            if NearZero(thetadotlist[i]):
                thetadotlist[i] = 0
        joints_violated = testJointLimits(thetalist, thetadotlist, dt)
        if len(joints_violated) != 0:
            for joint in joints_violated:
                J_e[:, joint+4] = 0
        else:
            violated = False
    return thetadotlist


def EndEffectorConfiguration(thetalist):
    """Compute the end effector configuration

    :param thetalist: youBot configuration

    :return: end effector configuration

    """
    phi = thetalist[0]
    x = thetalist[1]
    y = thetalist[2]
    T_sb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                     [np.sin(phi), np.cos(phi), 0, y],
                     [0, 0, 1, 0.0963],
                     [0, 0, 0, 1]])
    T_0e = FKinBody(M_0e, B_list, thetalist[3:8])
    T_se = T_sb @ T_b0 @ T_0e
    return T_se


def testJointLimits(thetalist, thetadotlist, dt):
    """Test to see if any joint limits violate the constraints

    :param thetalist: youBot configuration
    :param thetalist: youBot configuration velocities
    :param dt: time step

    :return: list of joint limits violated

    """
    arm_joint_angles = thetalist[3:8]
    arm_joint_speeds = thetadotlist[4:]
    joints_violated = []
    new_angles = np.array(arm_joint_angles) + dt * np.array(arm_joint_speeds)
    for i in range(len(new_angles)):
        if joint_limits[i] is not None:
            if new_angles[i] < joint_limits[i][0] or new_angles[i] > joint_limits[i][1]:
                joints_violated.append(i)

    return joints_violated


def MobileManipulatiion(cube_init, cube_final, youBot_init, T_se_init, Kp=np.zeros((4, 4)), Ki=np.zeros((4, 4))) -> None:
    """Perform full simulation of youBot."""
    T_sc_initial = np.array([[1, 0, 0,     cube_init[0]],
                             [0, 1, 0,     cube_init[1]],
                             [0, 0, 1,     cube_init[2]],
                             [0, 0, 0,                1]])

    T_sc_goal = np.array([[0, 1, 0,     cube_final[0]],
                          [-1, 0, 0,    cube_final[1]],
                          [0, 0, 1,     cube_final[2]],
                          [0, 0, 0,                 1]])
    thetalist = youBot_init
    traj = TrajectoryGenerator(
        T_se_init=T_se_init,
        T_sc_init=T_sc_initial,
        T_sc_final=T_sc_goal,
        T_ce_grasp=T_ce_grasp,
        T_ce_standoff=T_ce_standoff
    )

    T_se = EndEffectorConfiguration(thetalist)
    states = []
    states.append(thetalist)
    error = []
    for i in range(len(traj) - 1):
        X_d = np.array(
            [[traj[i][0], traj[i][1], traj[i][2], traj[i][9]],
             [traj[i][3], traj[i][4], traj[i][5], traj[i][10]],
             [traj[i][6], traj[i][7], traj[i][8], traj[i][11]],
             [0, 0, 0, 1]]
        )
        X_d_next = np.array(
            [[traj[i+1][0], traj[i+1][1], traj[i+1][2], traj[i+1][9]],
             [traj[i+1][3], traj[i+1][4], traj[i+1][5], traj[i+1][10]],
             [traj[i+1][6], traj[i+1][7], traj[i+1][8], traj[i+1][11]],
             [0, 0, 0, 1]]
        )
        V, X_err = FeedbackControl(
            X=T_se,
            X_d=X_d,
            X_d_next=X_d_next,
            Kp=Kp,
            Ki=Ki,
            dt=0.01
        )
        error.append(X_err)
        J_e = JacobianEndeffector(thetalist)
        thetadotlist = GetJointSpeeds(V, J_e, thetalist, dt)
        new_state = NextState(thetalist, thetadotlist, dt=0.01, max_vel=50)
        new_state.append(traj[i+1][12])
        states.append(new_state)
        thetalist = new_state
        T_se = EndEffectorConfiguration(thetalist)

    return states, error


def MatToVec(T_list, gripper_state):
    """ Helper function used to convert the the trajectory from a matrix into a vector."""
    thetalist = []
    for T in T_list:
        thetalist.append([T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2],
                         T[2][0], T[2][1], T[2][2], T[0][3], T[1][3], T[2][3], gripper_state])

    return thetalist


def WriteCSV(traj, filename):
    """Helper function used to write the csv files."""
    f = open(filename, 'w')

    for T in traj:
        f_out = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % (
            T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11], T[12])
        f.write(f_out)


def WriteError(error, filename):
    """Write error to data file."""
    with open(filename, 'w', newline='') as file:
        for err in error:
            file.write(' '.join(map(str, err)) + '\n')


def GenerateErrorPlot(error):
    """Generate error plot over time."""
    t = np.linspace(0, len(error)*0.01, len(error))
    plt.plot(t, error)
    plt.title('error vs. time')
    plt.xlabel('time (s)')
    plt.ylabel('X_err')
    plt.legend([r'$\omega_x$', r'$\omega_y$',
               r'$\omega_z$', r'v_x', r'v_y', r'v_z'])
    plt.show()


def main():
    """Main function."""
    logging.basicConfig(filename='overshoot_log.txt', level=logging.INFO)
    logging.info('$ python3 code/overshoot.py')
    cube_init = cube_pos_i
    cube_final = cube_pos_f
    youBot_init = [1.0, -0.2, 0.1, 0, -0.149,
                   -1.028, -0.707, 0, 0, 0, 0, 0, 0]

    T_se_init = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [-1, 0, 0, 0.5],
                          [0, 0, 0, 1]])
    Kp = np.array([[3.5, 0, 0, 0],
                   [0, 3.5, 0, 0],
                   [0, 0, 3.5, 0],
                   [0, 0, 0, 3.5]])
    Ki = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    states, error = MobileManipulatiion(
        cube_init=cube_init,
        cube_final=cube_final,
        youBot_init=youBot_init,
        T_se_init=T_se_init,
        Kp=Kp,
        Ki=Ki
    )

    logging.info('Generating animation csv file.')
    WriteCSV(states, 'overshoot.csv')

    logging.info('Writing error plot data.')
    WriteError(error, 'overshoot_error_data.txt')
    GenerateErrorPlot(error)

    logging.info('Done.')


if __name__ == '__main__':
    main()
