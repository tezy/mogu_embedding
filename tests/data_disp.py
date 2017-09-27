import os
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/home/tze'
file = 'attr_key_counts.txt'


def attr_key_freq():
    attr_freq_list = []
    with open(os.path.join(data_dir, file), 'r') as f:
        for line in f:
            attr, freq = line.strip().split(':')
            attr_freq_list.append(int(freq))

    x = np.arange(len(attr_freq_list))
    y = np.array(attr_freq_list)

    # plot with various axes scales
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)

    # linear
    ax = axs[0, 0]
    ax.plot(x, y)
    ax.set_yscale('linear')
    ax.set_title('linear')
    ax.grid(True)

    # log
    ax = axs[0, 1]
    ax.plot(x, y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('dual log')
    ax.grid(True)

    ax = axs[1, 0]
    ax.plot(x, y)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_title('x: linear, y: log')
    ax.grid(True)

    ax = axs[1, 1]
    ax.plot(x, y)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_title('x: log, y: linear')
    ax.grid(True)


    # # symmetric log
    # ax = axs[1, 1]
    # ax.plot(x, y - y.mean())
    # ax.set_yscale('symlog', linthreshy=0.02)
    # ax.set_title('symlog')
    # ax.grid(True)
    #
    # # logit
    # ax = axs[1, 0]
    # ax.plot(x, y)
    # ax.set_yscale('logit')
    # ax.set_title('logit')
    # ax.grid(True)
    # ax.yaxis.set_minor_formatter(NullFormatter())

    plt.show()


if __name__ ==  '__main__':
    attr_key_freq()