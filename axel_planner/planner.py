import numpy as np
import matplotlib.pyplot as plt

import time

from copy import deepcopy


def quintic_func(q0, qf, T, qd0=0, qdf=0):
    """
    Quintic scalar polynomial as a function

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param T: trajectory time
    :type T: float
    :param qd0: initial velocity, defaults to 0
    :type q0: float, optional
    :param qdf: final velocity, defaults to 0
    :type q0: float, optional
    :return: polynomial function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :rtype: callable

    Returns a function which computes the specific quintic polynomial, and its
    derivatives, as described by the parameters.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import quintic_func
        >>> f = quintic_func(1, 2, 5)
        >>> f(0)
        >>> f(5)
        >>> f(2.5)

    :seealso: :func:`quintic` :func:`trapezoidal_func`
    """

    # solve for the polynomial coefficients using least squares
    # fmt: off
    X = [
        [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
        [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
        [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
        [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
        [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
    ]
    # fmt: on
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )


def get_trajectory(
    start_point_xy, end_point_xy, traj_velocity=0.5, traj_point_per_meter=100
):
    traj_length = np.linalg.norm(np.array(end_point_xy) - np.array(start_point_xy))

    time_traj = traj_length / traj_velocity
    num_points = int(traj_length * traj_point_per_meter)
    traj_dt = time_traj / num_points

    x_fun_traj = quintic_func(start_point_xy[0], end_point_xy[0], time_traj)
    y_fun_traj = quintic_func(start_point_xy[1], end_point_xy[1], time_traj)

    time_array = np.linspace(0, time_traj, num_points)
    x_traj = np.array([x_fun_traj(t)[0] for t in time_array])
    y_traj = np.array([y_fun_traj(t)[0] for t in time_array])

    return x_traj, y_traj, time_traj, num_points, traj_dt

