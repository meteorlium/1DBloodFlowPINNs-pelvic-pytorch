import torch
import torch.nn as nn


class Pinn(nn.Module):
    """physics informed neural Networks.

    cite: Raissi, M., P. Perdikaris, and G. E. Karniadakis. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378 (February 1, 2019): 686–707. https://doi.org/10.1016/j.jcp.2018.10.045.

    Args:
        net (nn): a full connected net with xavier_normal_ init weights.
    """

    def __init__(self):
        super(Pinn, self).__init__()
        # layer_parameter = [2, 100, 100, 100, 100, 100, 100, 100, 3]
        self.layer1 = nn.Linear(2, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.layer5 = nn.Linear(100, 100)
        self.layer6 = nn.Linear(100, 100)
        self.layer7 = nn.Linear(100, 100)
        self.layer8 = nn.Linear(100, 3)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        x = torch.tanh(self.layer6(x))
        x = torch.tanh(self.layer7(x))
        x = self.layer8(x)
        A, u, p = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return A.exp(), u, p  # ? exp?


def weights_init(m):
    """ init weights and biases of Pinn.

    Use method xavier normal for weights.
    Set biases to zeros.

    Args:
        m (Pinn): a Pinn model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


def predict(x, t, i_vessel, net=None, dtype=torch.float32, device="cpu"):
    """Use existing models and data (x, t) to predict (A, u, p)

    Args:
        x (np.array or tensor): input data
        t (np.array or tensor): input data
        i_vessel (int): number of vessel
        net (Pinn, optional): model. Defaults to None.
        dtype (,optional): type of torch data. Defaults to torch.float32.
        device (str, optional): type of torch data. Defaults to "cpu".

    Returns:
        A_pred, u_pred, p_pred (np.array): output of net[i_vessel](x,t)
    """
    import utility
    # get net
    if not net:
        import os
        path_model = "result_model"
        net_state_dict = torch.load(
            os.path.join(path_model, f'net_{i_vessel}.pt'))
        net = Pinn().to(device)
        net.load_state_dict(net_state_dict)
    # preprocess
    pre = utility.load_preprocess()
    x = pre.preprocess_x(torch.tensor(x, dtype=dtype, device=device), i_vessel)
    t = pre.preprocess_t(torch.tensor(t, dtype=dtype, device=device))
    A_pred, u_pred, p_pred = net(torch.cat((x, t), dim=1))
    _, _, U, _, _, p0, A0 = utility.preprocess_parameter()
    # forward
    A_pred, u_pred, p_pred = net(torch.cat((x, t), dim=1))
    A_pred = A_pred.detach().cpu().numpy()*A0
    u_pred = u_pred.detach().cpu().numpy()*U
    p_pred = p_pred.detach().cpu().numpy()*p0

    return A_pred, u_pred, p_pred


if __name__ == "__main__":
    net = Pinn()
    net.apply(weights_init)
    print(net)
    input = torch.tensor(
        [[1., 2.],
         [2., 3.],
         [3., 4.]])
    a, u, p = net(input)
    print(a)
    print(u)
    print(p)
