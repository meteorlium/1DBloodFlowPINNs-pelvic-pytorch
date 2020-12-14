import torch


class DatasetMeasurement(torch.utils.data.Dataset):
    """dataset used in training Pinn, to calculate measurement loss
    """

    def __init__(self, x_measurement, t_measurement,
                 A_training, u_training, dtype=torch.float32, device='cpu'):
        self.x_measurement = torch.tensor(
            x_measurement, dtype=dtype, device=device)
        self.t_measurement = torch.tensor(
            t_measurement, dtype=dtype, device=device)
        self.A_training = torch.tensor(A_training, dtype=dtype, device=device)
        self.u_training = torch.tensor(u_training, dtype=dtype, device=device)

    def __getitem__(self, index):
        # x, t, A, u match in measurement data
        x = self.x_measurement[index]
        t = self.t_measurement[index]
        A = self.A_training[index]
        u = self.u_training[index]
        return x, t, A, u

    def __len__(self):
        return len(self.x_measurement)


class DatasetResidual(torch.utils.data.Dataset):
    """dataset used in training Pinn, to calculate residual loss
    """

    def __init__(self, x_residual, t_residual, dtype=torch.float32, device='cpu'):
        self.x_residual = torch.tensor(x_residual, dtype=dtype, device=device)
        self.t_residual = torch.tensor(t_residual, dtype=dtype, device=device)

    def __getitem__(self, index):
        # x, t match in residual data
        x = self.x_residual[index]
        t = self.t_residual[index]
        return x, t

    def __len__(self):
        return len(self.x_residual)


class DatasetInterface(torch.utils.data.Dataset):
    """dataset used in training Pinn, to calculate interface loss
    """

    def __init__(self, x1, x2, x3, t, dtype=torch.float32, device='cpu'):
        # t_interface = t_residual data
        self.t = torch.tensor(t, dtype=dtype, device=device)
        size = t.shape
        self.x1 = torch.ones(size=size, dtype=dtype, device=device)*x1
        self.x2 = torch.ones(size=size, dtype=dtype, device=device)*x2
        self.x3 = torch.ones(size=size, dtype=dtype, device=device)*x3

    def __getitem__(self, index):
        t = self.t[index]
        x1 = self.x1[index]
        x2 = self.x2[index]
        x3 = self.x3[index]
        return x1, x2, x3, t

    def __len__(self):
        return len(self.t)


if __name__ == "__main__":
    from utility import load_data
    [x_measurement, A_training, u_training, x_residual, t_residual,
        t_measurement, bif_points, test_points, t_initial] = load_data(200)

    print(x_residual[0][:8])
    print(t_residual[:8])

    residual_dataset = DatasetResidual(x_residual[0][:8], t_residual[:8])
    residual_dataloader = torch.utils.data.DataLoader(
        dataset=residual_dataset,
        batch_size=5,
        shuffle=True
    )
    # for i in range(3):
    #     print(f"epoch :{i}")
    #     for x, t in residual_dataloader:
    #         print("x:", x)
    #         print("t:", t)
    for _ in range(5):
        data = next(iter(residual_dataloader))
        print(data)
