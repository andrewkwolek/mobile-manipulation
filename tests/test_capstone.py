import numpy as np
import best as mm

X_d = np.array([[0, 0, 1, 0.5],
                [0, 1, 0, 0],
                [-1, 0, 0, 0.5],
                [0, 0, 0, 1]])

X_d_next = np.array([[0, 0, 1, 0.6],
                     [0, 1, 0, 0],
                     [-1, 0, 0, 0.3],
                     [0, 0, 0, 1]])

X = np.array([[0.170, 0, 0.985, 0.387],
              [0, 1, 0, 0],
              [-0.985, 0, 0.170, 0.570],
              [0, 0, 0, 1]])

Kp = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])


def test_feedback_control():
    V, X_err = mm.FeedbackControl(X=X, X_d=X_d, X_d_next=X_d_next)
    assert V[0] == 0
    assert V[1] == 0
    assert V[2] == 0
    assert V[3] == 21.4
    assert V[4] == 0
    assert np.round(V[5], 3) == 6.45

    assert X_err[0] == 0
    assert np.round(X_err[1], 3) == 0.171
    assert X_err[2] == 0
    assert np.round(X_err[3], 2) == 0.080
    assert X_err[4] == 0
    assert np.round(X_err[5], 3) == 0.107

    V, X_err = mm.FeedbackControl(X=X, X_d=X_d, X_d_next=X_d_next, Kp=Kp)
    assert V[0] == 0
    assert np.round(V[1], 3) == 0.171
    assert V[2] == 0
    assert np.round(V[3], 2) == 21.48
    assert V[4] == 0
    assert np.round(V[5], 2) == 6.56


def test_jacobian():
    q = [0, 0, 0, 0, 0, 0.2, -1.6, 0]
    J_e = mm.JacobianEndeffector(q)

    assert np.round(J_e[0][0], 3) == 0.030
    assert np.round(J_e[0][1], 3) == -0.030
    assert np.round(J_e[0][2], 3) == -0.030
    assert np.round(J_e[0][3], 3) == 0.030
    assert np.round(J_e[0][4], 3) == -0.985
    assert np.round(J_e[0][5], 3) == 0.0
    assert np.round(J_e[0][6], 3) == 0.0
    assert np.round(J_e[0][7], 3) == 0.0
    assert np.round(J_e[0][8], 3) == 0.0

    assert np.round(J_e[1][0], 3) == 0.0
    assert np.round(J_e[1][1], 3) == 0.0
    assert np.round(J_e[1][2], 3) == 0.0
    assert np.round(J_e[1][3], 3) == 0.0
    assert np.round(J_e[1][4], 3) == 0.0
    assert np.round(J_e[1][5], 3) == -1
    assert np.round(J_e[1][6], 3) == -1
    assert np.round(J_e[1][7], 3) == -1
    assert np.round(J_e[1][8], 3) == 0.0

    assert np.round(J_e[2][0], 3) == -0.005
    assert np.round(J_e[2][1], 3) == 0.005
    assert np.round(J_e[2][2], 3) == 0.005
    assert np.round(J_e[2][3], 3) == -0.005
    assert np.round(J_e[2][4], 3) == 0.170
    assert np.round(J_e[2][5], 3) == 0
    assert np.round(J_e[2][6], 3) == 0
    assert np.round(J_e[2][7], 3) == 0
    assert np.round(J_e[2][8], 3) == 1.0

    assert np.round(J_e[3][0], 3) == 0.002
    assert np.round(J_e[3][1], 3) == 0.002
    assert np.round(J_e[3][2], 3) == 0.002
    assert np.round(J_e[3][3], 3) == 0.002
    assert np.round(J_e[3][4], 3) == 0.0
    assert np.round(J_e[3][5], 3) == -0.240
    assert np.round(J_e[3][6], 3) == -0.214
    assert np.round(J_e[3][7], 3) == -0.218
    assert np.round(J_e[3][8], 3) == 0

    assert np.round(J_e[4][0], 3) == -0.024
    assert np.round(J_e[4][1], 3) == 0.024
    assert np.round(J_e[4][2], 3) == 0.0
    assert np.round(J_e[4][3], 3) == 0.0
    assert np.round(J_e[4][4], 3) == 0.221
    assert np.round(J_e[4][5], 3) == 0
    assert np.round(J_e[4][6], 3) == 0
    assert np.round(J_e[4][7], 3) == 0
    assert np.round(J_e[4][8], 3) == 0

    assert np.round(J_e[5][0], 3) == 0.012
    assert np.round(J_e[5][1], 3) == 0.012
    assert np.round(J_e[5][2], 3) == 0.012
    assert np.round(J_e[5][3], 3) == 0.012
    assert np.round(J_e[5][4], 3) == 0.0
    assert np.round(J_e[5][5], 3) == -0.288
    assert np.round(J_e[5][6], 3) == -0.135
    assert np.round(J_e[5][7], 3) == 0
    assert np.round(J_e[5][8], 3) == 0


def test_speeds():
    q = [0, 0, 0, 0, 0, 0.2, -1.6, 0]
    J_e = mm.JacobianEndeffector(q)
    V, X_err = mm.FeedbackControl(X=X, X_d=X_d, X_d_next=X_d_next)
    thetadotlist = mm.GetJointSpeeds(V, J_e)

    assert np.round(thetadotlist[0], 1) == 157.1
    assert np.round(thetadotlist[1], 1) == 157.1
    assert np.round(thetadotlist[2], 1) == 157.1
    assert np.round(thetadotlist[3], 1) == 157.1
    assert np.round(thetadotlist[4], 1) == 0
    assert np.round(thetadotlist[5], 1) == -652.6
    assert np.round(thetadotlist[6], 2) == 1398.04
    assert np.round(thetadotlist[7], 1) == -745.4
    assert np.round(thetadotlist[8], 1) == 0

    V, X_err = mm.FeedbackControl(X=X, X_d=X_d, X_d_next=X_d_next, Kp=Kp)
    thetadotlist = mm.GetJointSpeeds(V, J_e)
    assert np.round(thetadotlist[0], 1) == 157.4
    assert np.round(thetadotlist[1], 1) == 157.4
    assert np.round(thetadotlist[2], 1) == 157.4
    assert np.round(thetadotlist[3], 1) == 157.4
    assert np.round(thetadotlist[4], 1) == 0
    assert np.round(thetadotlist[5], 1) == -654
    assert np.round(thetadotlist[6], 2) == 1400.3
    assert np.round(thetadotlist[7], 1) == -746.5
    assert np.round(thetadotlist[8], 1) == 0


def test_next_state():
    thetalist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    thetadotlist = [10, 10, 10, 10, 0, 0, 0, 0, 0]
    traj = []
    traj.append(thetalist)
    for i in range(100):
        thetalist = mm.NextState(thetalist, thetadotlist, 0.01, 20)
        traj.append(thetalist)

    assert np.round(traj[-1][1], 3) == 0.475

    thetadotlist = [-10, 10, -10, 10, 0, 0, 0, 0, 0]
    traj = []
    traj.append(thetalist)
    for i in range(100):
        thetalist = mm.NextState(thetalist, thetadotlist, 0.01, 20)
        traj.append(thetalist)

    assert np.round(traj[-1][2], 3) == 0.475

    thetadotlist = [-10, 10, 10, -10, 0, 0, 0, 0, 0]
    traj = []
    traj.append(thetalist)
    for i in range(100):
        thetalist = mm.NextState(thetalist, thetadotlist, 0.01, 20)
        traj.append(thetalist)

    assert np.round(traj[-1][0], 3) == 1.234
