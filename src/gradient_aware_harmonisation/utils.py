import matplotlib.pyplot as plt
from typing import Tuple

def compute_gradient(seq_xy: Tuple[list, list]) -> list:
    gradient = []
    for i in range(len(seq_xy[0])):
        if i == 0 or i == len(seq_xy[0])-1:
            gradient.append(None)
        else:
            numerator=seq_xy[1][i]-seq_xy[1][i-1]
            denominator=seq_xy[0][i]-seq_xy[0][i-1]
            gradient.append(numerator/denominator)
    return gradient

def plot_funcs(f1, f2, df1, df2, x0):
    plt.figure(figsize=(6,3))
    plt.plot(f1[0], f1[1], color='blue', label='f1')
    plt.plot(f2[0], f2[1], color='orange', label='f2')
    plt.plot(f1[0], df1, color="blue", linestyle="dotted", label="df1")
    plt.plot(f2[0], df2, color="orange", linestyle="dotted", label="df2")
    plt.legend(ncol=2, handlelength=0.8, columnspacing=0.8)
    plt.axvline(x0, color="red")
    plt.show()