import torch
import torch.nn as nn

from utility import preprocess_parameter

n_vessel = 7


# def sse(x, y=0):
#     return (x-y).pow(2).sum()


def mse(x, y=0):
    return (x-y).pow(2).mean()


class PinnLoss(nn.Module):
    def __init__(self, x_std, t_std, dtype=torch.float32, device="cpu"):

        A_0, beta, U, _, _, p0, A0 = preprocess_parameter()
        self.A_0 = A_0
        self.beta = beta
        self.U = U
        self.p0 = p0
        self.A0 = A0

        self.x_std = torch.tensor(x_std, dtype=dtype, device=device)
        self.t_std = torch.tensor(t_std, dtype=dtype, device=device)

    def loss_interface(self, A1, u1, p1, A2, u2, p2, A3, u3, p3):
        """interface loss of a batch points in vessel 1, vessel 2, and vessel 3

        Args:
            A1 (tensor): Area of vessel 1
            u1 (tensor): velocity of vessel 1
            p1 (tensor): pressure of vessel 1
            A2 (tensor): Area of vessel 2
            u2 (tensor): velocity of vessel 2
            p2 (tensor): pressure of vessel 2
            A3 (tensor): Area of vessel 3
            u3 (tensor): velocity of vessel 3
            p3 (tensor): pressure of vessel 3

        Returns:
            (float): interface loss of a batch inter points
        """
        Q1 = A1 * u1
        Q2 = A2 * u2
        Q3 = A3 * u3
        loss_mass = mse(Q1, Q2+Q3)
        p1 = p1 + 0.5*u1**2
        p2 = p2 + 0.5*u2**2
        p3 = p3 + 0.5*u3**2
        loss_momentum = mse(p1, p2) + mse(p1, p3)
        return loss_mass + loss_momentum

    def loss_residual(self, A_pred, u_pred, p_pred, p_x, A_t, A_x, u_t, u_x, i_vessel):
        """residual loss of a batch points in i_vessel

        Args:
            A_pred (tensor): Area of vessel
            u_pred (tensor): velocity of vessel
            p_pred (tensor): pressure of vessel
            p_x (tensor): partial p / partial x
            A_t (tensor): partial A / partial t
            A_x (tensor): partial A / partial x
            u_t (tensor): partial u / partial t
            u_x (tensor): partial u / partial x
            i_vessel (int): number of vessel where the points in

        Returns:
            (float): residual loss of a batch points
        """
        beta = self.beta[i_vessel]
        A0 = self.A0
        A_0 = self.A_0[i_vessel]
        p0 = self.p0
        x_std = self.x_std[i_vessel]
        t_std = self.t_std

        p_x /= x_std
        A_t /= t_std
        A_x /= x_std
        u_t /= t_std
        u_x /= x_std

        r_A = A_t + u_pred*A_x + A_pred*u_x
        r_u = u_t + p_x + u_pred*u_x
        r_p = beta * ((A_pred * A0).sqrt() - A_0**0.5)

        loss_rA = mse(r_A)
        loss_ru = mse(r_u)
        loss_rp = mse(p_pred, r_p/p0)
        return loss_rA + loss_ru + loss_rp

    def loss_measurement(self, A_pred, u_pred, A_training, u_training):
        """measurement loss of a batch points in i_vessel

        Args:
            A_pred(tensor): Area of vessel
            u_pred (tensor): velocity of vessel
            A_training (tensor): area data for training
            u_training (tensor): velocity data for training

        Returns:
            (float): measurement loss of a batch points
        """
        A0 = self.A0
        U = self.U
        loss_A = mse(A_pred, A_training/A0)
        loss_u = mse(u_pred, u_training/U)
        return loss_A + loss_u


if __name__ == '__main__':
    import torch
    x_std = torch.ones(n_vessel)
    t_std = torch.tensor(1.)

    Loss = PinnLoss(x_std=x_std, t_std=t_std)
    # interface_loss(self, A1, u1, p1, A2, u2, p2, A3, u3, p3)
    int_loss = Loss.loss_interface(
        torch.tensor(1.),
        torch.tensor(2.),
        torch.tensor(3.),
        torch.tensor(4.),
        torch.tensor(5.),
        torch.tensor(6.),
        torch.tensor(7.),
        torch.tensor(8.),
        torch.tensor(9.))
    print(f"int_loss : {int_loss}")
    # residual_loss(self, A_pred, u_pred, p_pred, p_x, A_t, A_x, u_t, u_x, i_vessel)
    res_loss = Loss.loss_residual(
        torch.tensor(1.),
        torch.tensor(2.),
        torch.tensor(3.),
        torch.tensor(4.),
        torch.tensor(5.),
        torch.tensor(6.),
        torch.tensor(7.),
        torch.tensor(8.),
        1)
    print(f"res_loss : {res_loss}")
    # measurement_loss(self, A_pred, u_pred, A_training, u_training)
    mea_loss = Loss.loss_measurement(
        torch.tensor(1.),
        torch.tensor(2.),
        torch.tensor(3.),
        torch.tensor(4.))
    print(f"mea_loss : {mea_loss}")
