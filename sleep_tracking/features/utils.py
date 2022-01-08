import numpy as np
from scipy import interpolate


def smooth_moving_average(y, window):
    box = np.ones(window) / window
    if window > 1:
        y = np.insert(y, 0, np.flip(y[0:int(window / 2)]))  # Pad by repeating boundary conditions
        y = np.insert(y, len(y) - 1, np.flip(y[int(-window / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    if window % 2 == 0:
        y_smooth = y_smooth[:-1]

    return y_smooth


def interpolate_and_smooth(xfull, yfull, xinterp, window):
    yinterp = interpolate.interp1d(x=xfull, y=yfull, kind='cubic', fill_value="extrapolate")(xinterp)
    yinterp = smooth_moving_average(yinterp, window)
    return yinterp


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value


def convolve_with_dog(y, box_pts):
    # y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth
