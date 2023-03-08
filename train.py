import torch
import numpy as np
import os

from utility import load_data, preprocess_parameter, Preprocess, draw
from net import Pinn, weights_init
from loss import PinnLoss
from dataset import DatasetInterface, DatasetMeasurement, DatasetResidual
from torch.utils.data import DataLoader

device = torch.device('cuda')
dtype = torch.float32


if __name__ == "__main__":

    # * load data

    n_residual = 2000  # Total number of residual points
    [x_measurement, A_training, u_training, x_residual, t_residual,
        t_measurement, bif_points, test_points, t_initial] = load_data(n_residual)

    # x_measurement[0].shape = (1748, 1)
    # A_training[0].shape = (1748, 1)
    # u_training[0].shape = (1748, 1)
    # t_measurement.shape = (1748, 1)
    # x_residual[0].shape = (2000, 1)
    # t_residual.shape = (2000, 1)
    # t_initial.shape = (874, 1)
    # bif_points = np.array, len = 3
    # test_points = np.array, len = 7

    n_vessel = len(x_measurement)

    # * preprocess data

    _, _, _, L, T, _, _ = preprocess_parameter()
    x_mean = [(x/L).mean() for x in x_residual]
    x_std = [(x/L).std() for x in x_residual]
    t_mean = (t_residual/T).mean()
    t_std = (t_residual/T).std()
    pre = Preprocess(x_mean, x_std, t_mean, t_std, L, T)
    pre.save_parameters()

    for i in range(n_vessel):
        x_measurement[i] = pre.preprocess_x(x_measurement[i], i)
        x_residual[i] = pre.preprocess_x(x_residual[i], i)
    t_residual = pre.preprocess_t(t_residual)
    t_measurement = pre.preprocess_t(t_measurement)
    t_initial = pre.preprocess_t(t_initial)

    # * dataset for network

    BATCH_SIZE = 1024  # batch num of residual
    no_bound_vessel_id = [2, 6]  # vessel 2 and 6 don't have boundary data
    dataloader_measurement = []
    dataloader_residual = []
    for i in range(n_vessel):
        if i+1 in no_bound_vessel_id:
            t_mea = t_initial
        else:
            t_mea = t_measurement
        dataset_measurement = DatasetMeasurement(
            x_measurement=x_measurement[i],
            t_measurement=t_mea,
            A_training=A_training[i],
            u_training=u_training[i],
            dtype=dtype,
            device=device
        )
        dataloader_measurement.append(DataLoader(
            dataset=dataset_measurement,
            batch_size=len(dataset_measurement)
        ))
        dataset_residual = DatasetResidual(
            x_residual=x_residual[i],
            t_residual=t_residual,
            dtype=dtype,
            device=device
        )
        dataloader_residual.append(DataLoader(
            dataset=dataset_residual,
            batch_size=BATCH_SIZE,
            shuffle=True
        ))

    # interface data
    id_interface = [[3, 6, 2], [6, 7, 5], [2, 1, 4]]
    dataloader_interface = []
    for id, point in zip(id_interface, bif_points):
        x_0 = pre.preprocess_x(point, id[0]-1)
        x_1 = pre.preprocess_x(point, id[1]-1)
        x_2 = pre.preprocess_x(point, id[2]-1)
        dataset_interface = DatasetInterface(
            x1=x_0,
            x2=x_1,
            x3=x_2,
            t=t_residual,
            dtype=dtype,
            device=device
        )  # data t_interface = data t_residual
        dataloader_interface.append(DataLoader(
            dataset=dataset_interface,
            batch_size=BATCH_SIZE,
            shuffle=True
        ))

    # * initial network weights and parameters

    net = [Pinn().to(device).apply(weights_init) for _ in range(n_vessel)]
    epoch = 290000 + 50000
    lr = 1e-3
    params = [{"params": net_i.parameters()} for net_i in net]
    optimizer = torch.optim.Adam(params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=290000, gamma=0.1)
    # train: epoch,lr = 290000,1e-3 then 50000,1e-4

    epoch_printloss = 100
    epoch_draw = 1000
    epoch_recordloss = 100
    epoch_save = 1000

    loss_fn = PinnLoss(x_std=pre.x_std,
                       t_std=pre.t_std,
                       dtype=dtype,
                       device=device)
    loss_fn_interface = loss_fn.loss_interface
    loss_fn_residual = loss_fn.loss_residual
    loss_fn_measurement = loss_fn.loss_measurement

    # * train

    # record loss
    loss_list = []
    loss_measurement_list = []
    loss_residual_list = []
    loss_interface_list = []

    for i_epoch in range(epoch):
        # * forward

        loss_measurement = 0
        loss_residual = 0
        loss_interface = 0

        # measurement
        for i_vessel in range(n_vessel):
            x_batch, t_batch, A_batch, u_batch = next(
                iter(dataloader_measurement[i_vessel]))
            A_pred, u_pred, _ = net[i_vessel](
                torch.cat((x_batch, t_batch), dim=1).to(device))
            loss_measurement += loss_fn_measurement(
                A_pred, u_pred, A_batch, u_batch)

        # residual
        for i_vessel in range(n_vessel):
            x_batch, t_batch = next(iter(dataloader_residual[i_vessel]))
            x_batch.requires_grad = True
            t_batch.requires_grad = True

            A_pred, u_pred, p_pred = net[i_vessel](
                torch.cat((x_batch, t_batch), dim=1).to(device))

            p_x = torch.autograd.grad(
                p_pred.sum(), x_batch, retain_graph=True, create_graph=True)[0]
            A_x = torch.autograd.grad(
                A_pred.sum(), x_batch, retain_graph=True, create_graph=True)[0]
            A_t = torch.autograd.grad(
                A_pred.sum(), t_batch, retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(
                u_pred.sum(), x_batch, retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(
                u_pred.sum(), t_batch, retain_graph=True, create_graph=True)[0]

            loss_residual += loss_fn_residual(
                A_pred, u_pred, p_pred, p_x, A_t, A_x, u_t, u_x, i_vessel)

        # interface
        for i_interface in range(len(id_interface)):
            x_batch_1, x_batch_2, x_batch_3, t_batch = next(
                iter(dataloader_interface[i_interface]))
            A_pred_1, u_pred_1, p_pred_1 = net[id_interface[i_interface][0]-1](
                torch.cat((x_batch_1, t_batch), dim=1).to(device))
            A_pred_2, u_pred_2, p_pred_2 = net[id_interface[i_interface][1]-1](
                torch.cat((x_batch_2, t_batch), dim=1).to(device))
            A_pred_3, u_pred_3, p_pred_3 = net[id_interface[i_interface][2]-1](
                torch.cat((x_batch_3, t_batch), dim=1).to(device))
            loss_interface += loss_fn_interface(
                A_pred_1, u_pred_1, p_pred_1,
                A_pred_2, u_pred_2, p_pred_2,
                A_pred_3, u_pred_3, p_pred_3
            )
        loss = loss_measurement + loss_residual + loss_interface

        # * backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # * record and save results

        if i_epoch % epoch_printloss == 0:
            print(
                f"i_epoch: {i_epoch:6d}",
                f"loss: {loss:.3e}",
                f"loss_m: {loss_measurement:.3e}",
                f"loss_r: {loss_residual:.3e}",
                f"loss_i: {loss_interface:.3e}",
                sep=", ")

        if i_epoch % epoch_draw == 0:
            draw(test_points=test_points, net=net, i_draw=i_epoch //
                 epoch_draw, dtype=dtype, device=device)

        if i_epoch % epoch_recordloss == 0:
            loss_list.append(loss.detach().cpu())
            loss_measurement_list.append(loss_measurement.detach().cpu())
            loss_residual_list.append(loss_residual.detach().cpu())
            loss_interface_list.append(loss_interface.detach().cpu())

        if i_epoch % epoch_save == 0:
            # loss list
            path_save_loss = "result_loss"
            os.makedirs(path_save_loss, exist_ok=True)
            np.save(os.path.join(path_save_loss, "loss.npy"),
                    np.array(loss_list))
            np.save(os.path.join(path_save_loss, "loss_m.npy"),
                    np.array(loss_measurement_list))
            np.save(os.path.join(path_save_loss, "loss_r.npy"),
                    np.array(loss_residual_list))
            np.save(os.path.join(path_save_loss, "loss_i.npy"),
                    np.array(loss_interface_list))
            # weights of net
            for i in range(len(net)):
                torch.save(net[i].state_dict(), os.path.join(
                    "result_model", f"net_{i}.pt"))
    print("finish training!")
