import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path = "result_loss"
    epoch_recordloss = 100

    loss = np.load(os.path.join(path, 'loss.npy'), allow_pickle=True)[1:]
    loss_m = np.load(os.path.join(path, 'loss_m.npy'), allow_pickle=True)[1:]
    loss_r = np.load(os.path.join(path, 'loss_r.npy'), allow_pickle=True)[1:]
    loss_i = np.load(os.path.join(path, 'loss_i.npy'), allow_pickle=True)[1:]

    draw_x = list(range(len(loss)))

    fig = plt.figure(1, figsize=(9, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    linewidth = 0.4

    plt.subplot(221)
    loss_name = 'log( loss )'
    plt.plot(draw_x, np.log(loss), linewidth=linewidth)
    plt.xlabel(f'iter num * {epoch_recordloss}')
    plt.ylabel(loss_name)

    plt.subplot(222)
    loss_name = 'log( loss_measurement )'
    plt.plot(draw_x, np.log(loss_m), linewidth=linewidth)
    plt.xlabel(f'iter num * {epoch_recordloss}')
    plt.ylabel(loss_name)

    plt.subplot(223)
    loss_name = 'log( loss_residual )'
    plt.plot(draw_x, np.log(loss_r), linewidth=linewidth)
    plt.xlabel(f'iter num * {epoch_recordloss}')
    plt.ylabel(loss_name)

    plt.subplot(224)
    loss_name = 'log( loss_interface )'
    plt.plot(draw_x, np.log(loss_i), linewidth=linewidth)
    plt.xlabel(f'iter num * {epoch_recordloss}')
    plt.ylabel(loss_name)

    plt.suptitle('Training loss')

    plt.savefig(os.path.join(path, 'log_loss_4.jpg'))
    plt.close(1)
