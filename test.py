from net import predict
import torch

if __name__ == "__main__":

    # * input : x, t, i_vessel

    # i_vessel range: i_vessel in {0,1,2,3,4,5,6}
    i_vessel = 0
    # x range:
    # i_vessel = 0 : x in [0.08061722, 0.21500125]
    # i_vessel = 1 : x in [0.01068202, 0.08061722]
    # i_vessel = 2 : x in [0.        , 0.01068202]
    # i_vessel = 3 : x in [0.08061722, 0.21500125]
    # i_vessel = 4 : x in [0.07734581, 0.22684901]
    # i_vessel = 5 : x in [0.01068202, 0.07734581]
    # i_vessel = 6 : x in [0.07734581, 0.22508094]
    x = torch.tensor([[0.1]])

    # t range: t in [0, 2.1825]
    t = torch.tensor([[1.]])

    # you can input more (x, t) together,
    # e.g. x = torch.tensor([[0.1], [0.11], [0.12]])
    #      t = torch.tensor([[1.], [1.], [1.]])

    print(f"i_vessel = {i_vessel}")
    print(f"x = {x}")
    print(f"t = {t}")

    # * pred

    a, u, p = predict(
        x=x,
        t=t,
        i_vessel=i_vessel,
        net=None,  # load from "result_model"
        dtype=torch.float32,
        device="cpu"
    )

    print("pred A (area):")
    print(a)
    print("pred u (velocity):")
    print(u)
    print("pred p (pressure):")
    print(p)
