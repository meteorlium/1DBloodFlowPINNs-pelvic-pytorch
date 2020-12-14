import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

n_vessel = 7


def load_velocity_area_measurements():
    """load velocity and area measurements data from "test_vessel_i.npy".

    Returns:
        velocity_measurements (list of np.array): velocity measurements data of vessels.
        area_measurements (list of np.array): area measurements data of vessels.
    """
    velocity_measurement = []
    area_measurement = []
    for i in range(n_vessel):
        # data from "test_vessel_i.npy" which are equal to "(output/input_)vessel_i" in "Pelvic.py"
        test_vessel_path = "test_vessel_" + str(i+1) + ".npy"
        test_vessel_path = os.path.join("data", test_vessel_path)
        test_vessel = np.load(test_vessel_path, allow_pickle=True).item()
        velocity_measurement.append(test_vessel["Velocity"][:, None])
        area_measurement.append(test_vessel["Area"][:, None])
    return velocity_measurement, area_measurement


def load_t():
    """load t data from "test_vessel_3.npy".

    Returns:
        t (np.array): t data.
    """
    test_vessel = np.load(os.path.join(
        "data", "test_vessel_3.npy"), allow_pickle=True).item()
    t = test_vessel["Time"][:, None]
    t = t - t.min(0)
    return t


def set_x_measurement_residual(n_measurement, n_residual, no_bound_vessel_id):
    """get x_measurement data and x_residual data.

    Vessel[no_init_vessel_id] don't have boundary points. Their measurement data are (x_initial, t_initial).
    Other vessels have boundary points. Their measurement data are ([x_initial, x_boundary], [t_initial, t_boundary]).
    Test points are used to compare the results.
    Bif points are the interface points.


    Args:
        n_measurement (int): total number of measurement points.
        n_residual (int): total number of residual points.
        no_bound_vessel_id (list): id of vessels which do not have init data.

    Returns:
        x_measurement (list of np.array): x measurement data.
        x_residual (list of np.array): x residual data.
        bif_points (np.array): x of interface points.
        test_points (np.array): x of test points in each vessels.
    """

    # vessel shape:
    #            2 -> 4 and 1
    # 3 -> 6 and 2
    #      6 -> 5 and 7

    lower_bound = np.array([
        0.01068202 + 0.0699352,
        0.01068202,
        0.0,
        0.01068202 + 0.0699352,
        0.01068202 + 0.06666379,
        0.01068202,
        0.01068202 + 0.06666379
    ])
    upper_bound = np.array([
        0.01068202 + 0.0699352 + 0.13438403,
        0.01068202 + 0.0699352,
        0.01068202,
        0.01068202 + 0.0699352 + 0.13438403,
        0.01068202 + 0.06666379 + 0.1495032,
        0.01068202 + 0.06666379,
        0.01068202 + 0.06666379 + 0.14773513
    ])
    measurement_points = np.array([
        0.14882781,
        0.04564962,
        0.00534101,
        0.14780923,
        0.15209741,
        0.04401392,
        0.15121337
    ])
    test_points = np.array([
        upper_bound[1-1],
        0.06,
        lower_bound[3-1],
        upper_bound[4-1],
        upper_bound[5-1],
        0.03,
        upper_bound[7-1]
    ])
    bif_points = np.array([
        upper_bound[3-1],
        upper_bound[2-1],
        upper_bound[6-1]
    ])

    x_measurement = []
    for i in range(n_vessel):
        x_initial = np.linspace(
            lower_bound[i], upper_bound[i], n_measurement)[:, None]
        if i+1 in no_bound_vessel_id:
            x_measurement.append(np.vstack((x_initial)))
        else:
            x_boundary = measurement_points[i] * \
                np.ones(n_measurement)[:, None]
            x_measurement.append(
                np.vstack((x_initial, x_boundary))
            )

    x_residual = []
    for i in range(n_vessel):
        x_residual.append(
            lower_bound[i] + (upper_bound[i] - lower_bound[i]) * np.random.random((n_residual))[:, None])

    return x_measurement, x_residual, bif_points, test_points


def set_t_measurement_residual(t, n_measurement, n_residual):
    """set t_measurement, t_residual and t_initial as training data.

    T initial is equal to low bound * np.ones.
    T boundary is equal to t.
    T residual is set randomly.

    Args:
        t (np.array): data t.
        n_measurement (int): total number of measurement points.
        n_residual (int): total number of residual points.

    Returns:
        t_measurement (np.array): t_measurement data of vessel not in no_bound_vessel_id.
        t_residual (np.array): t residual data
        t_initial (np.array): t initial data of vessel in no_bound_vessel_id.
    """
    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)
    t_initial = lower_bound_t * np.ones((n_measurement))[:, None]
    t_boundary = t
    t_measurement = np.vstack((t_initial, t_boundary))
    t_residual = lower_bound_t + \
        (upper_bound_t-lower_bound_t) * \
        np.random.random((n_residual))[:, None]
    return t_measurement, t_residual, t_initial


def set_a_u_training(velocity_measurements, area_measurements, n_measurement, no_bound_vessel_id):
    """set a_training and u_training as training data.

    Training data of vessel in no_bound_vessel_id is different from other vessels.

    Args:
        velocity_measurements (list of np.array): velocity measurements data of vessels.
        area_measurements (list of np.array): area measurements data of vessels.
        n_measurement (int): total number of measurement points.
        no_init_vessel_id (list): id of vessels which do not have init data.

    Returns:
        a_training (list of np.array): data a for training.
        u_training (list of np.array): data u for training.
    """
    a_training = []
    u_training = []

    for i in range(n_vessel):
        a_initial = area_measurements[i][0, 0]*np.ones((n_measurement, 1))
        u_initial = velocity_measurements[i][0, 0]*np.ones((n_measurement, 1))
        if i+1 in no_bound_vessel_id:
            a_training.append(np.vstack((a_initial)))
            u_training.append(np.vstack((u_initial)))
        else:
            a_training.append(np.vstack((a_initial, area_measurements[i])))
            u_training.append(np.vstack((u_initial, velocity_measurements[i])))
    return a_training, u_training


def load_data(n_residual):
    """load training data from npy files

    Returns:
        x_measurement (list of np.array): x measurement data.
        a_training (list of np.array): data a for training.
        u_training (list of np.array): data u for training.
        x_residual (list of np.array): x residual data.
        t_residual (np.array): t residual data for training.
        t_measurement (np.array): t measurement data for training.
        bif_points (np.array): x of interface points.
        test_points (np.array): x of test points.
        t_initial (np.array): t initial data for training.
    """

    velocity_measurement, area_measurement = load_velocity_area_measurements()

    t = load_t()

    # equal to N_u in Pelvic.py, total number of measurement points while x = measurement point or t = 0
    n_measurement = t.shape[0]

    no_bound_vessel_id = [2, 6]

    x_measurement, x_residual, bif_points, test_points = set_x_measurement_residual(
        n_measurement, n_residual, no_bound_vessel_id)
    t_measurement, t_residual, t_initial = set_t_measurement_residual(
        t, n_measurement, n_residual)
    a_training, u_training = set_a_u_training(
        velocity_measurement, area_measurement, n_measurement, no_bound_vessel_id)
    return x_measurement, a_training, u_training, x_residual, t_residual, t_measurement, bif_points, test_points, t_initial


def preprocess_parameter():
    """init parameters for preprocess x and t.

        A_0, beta, U, p0, A0 are used in "loss.py" to calculate loss.
        A0, U, p0 are used to predict A, u, p.
        L, T are used to preprocess x and t.

    Returns:
        Omitted.
    """
    A_0 = [2.121382E-05,
           2.169600E-05,
           2.139937E-05,
           1.915712E-05,
           2.078009E-05,
           2.209141E-05,
           1.966408E-05]
    rho = 1060.
    beta = [26700411.41388087,
            26296966.62168744,
            26542712.62363585,
            28687170.14347438,
            27081607.95354047,
            25980955.92539433,
            28152317.1840086]
    U = 1e+1
    L = np.sqrt((1./7.)*sum(A_0))
    T = L / U
    p0 = rho * U ** 2
    A0 = L ** 2
    return A_0, beta, U, L, T, p0, A0


class Preprocess(object):
    """preprocess x and t.
    """

    def __init__(self, x_mean, x_std, t_mean, t_std, L, T):
        self.x_mean = x_mean
        self.x_std = x_std
        self.t_mean = t_mean
        self.t_std = t_std
        self.L = L
        self.T = T

    def non_dim_x(self, x):
        return x/self.L

    def non_dim_t(self, t):
        return t/self.T

    def standardize_x(self, x, i_vessel):
        return (x - self.x_mean[i_vessel])/self.x_std[i_vessel]

    def standardize_t(self, t):
        return (t - self.t_mean)/self.t_std

    def preprocess_x(self, x, i_vessel):
        x = self.non_dim_x(x)
        x = self.standardize_x(x, i_vessel)
        return x

    def preprocess_t(self, t):
        t = self.non_dim_t(t)
        t = self.standardize_t(t)
        return t

    def save_parameters(self):
        """save data to json file.
        """
        dic = {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "t_mean": self.t_mean,
            "t_std": self.t_std,
            "L": self.L,
            "T": self.T
        }
        file_js = open(os.path.join(
            "result_model", "pre_parameters.json"), "w")
        file_js.write(json.dumps(dic))
        file_js.close()


def load_preprocess():
    """set a Preprocess from "pre_parameters.json".

    Returns:
        pre ([class Preprocess]) : used to predict.
    """
    json_file = "result_model/pre_parameters.json"
    read_file = open(json_file, "r")
    data = json.load(read_file)
    pre = Preprocess(
        x_mean=data["x_mean"],
        x_std=data["x_std"],
        t_mean=data["t_mean"],
        t_std=data["t_std"],
        L=data["L"],
        T=data["T"]
    )
    return pre


def draw(test_points, net, i_draw, dtype, device):
    """Draw comparison charts.

    Args:
        test_points (np.array): x of test points.
        net (list of Pinn): Pinn model
        i_draw (int): draw number, used in figures
        dtype : type of torch data
        device : device of torch data
    """
    from net import predict

    # load velocity_test, pressure_test from vessel_i.npy
    velocity_test = []
    pressure_test = []
    for i in range(n_vessel):
        path = os.path.join("data", f"vessel_{i+1}.npy")
        vessel = np.load(path, allow_pickle=True).item()
        velocity_test.append(vessel["Velocity"][:, None])
        pressure_test.append(vessel["Pressure"][:, None])

    t = load_t()
    x = [point*np.ones(t.shape) for point in test_points]
    u_pred = []
    p_pred = []
    for i in range(n_vessel):
        _, u, p = predict(
            x=x[i],
            t=t,
            i_vessel=i,
            net=net[i],
            dtype=dtype,
            device=device
        )
        u_pred.append(u)
        p_pred.append(p)

    plt.figure(1, figsize=(22, 12), dpi=111,
               facecolor='w', frameon=False)
    for i in range(n_vessel):
        plt.subplot(241+i)
        plt.plot(t, u_pred[i], 'bo', linewidth=1, markersize=0.5,
                 label='Predicted velocity Vessel t_raw')
        plt.plot(t, velocity_test[i], 'ro', linewidth=1,
                 markersize=0.5,  label='Reference velocity Vessel')
        if i+1 in [1, 5]:
            plt.ylabel("Velocity in m/s")
        if i+1 in [2]:
            plt.legend(loc='upper center', bbox_to_anchor=(1.18, 1.23),
                       ncol=2, fancybox=True, shadow=True)
        if i+1 in [5]:
            plt.xlabel("t in s")
    plt.suptitle('Comparative velocity')
    plt.savefig(
        f"./result_Velocity/Comparative_Velocity_{i_draw:03}.jpg")
    plt.close(1)

    plt.figure(2, figsize=(22, 12), dpi=110,
               facecolor='w', frameon=False)
    for i in range(n_vessel):
        plt.subplot(241+i)
        plt.plot(t, p_pred[i], 'bo', linewidth=1, markersize=0.5,
                 label='Predicted pressure Vessel t_raw')
        plt.plot(t, pressure_test[i], 'ro', linewidth=1,
                 markersize=0.5,  label='Reference pressure Vessel')
        plt.ylim(6000, 15000)
        if i+1 in [1, 5]:
            plt.ylabel("Pressure in Pa")
        if i+1 in [2]:
            plt.legend(loc='upper center', bbox_to_anchor=(1.18, 1.23),
                       ncol=2, fancybox=True, shadow=True)
        if i+1 in [5]:
            plt.xlabel("t in s")
    plt.suptitle('Comparative pressure')
    plt.savefig(
        f"./result_Pressure/Comparative_Pressure_{i_draw:03}.jpg")
    plt.close(2)
