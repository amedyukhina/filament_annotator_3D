from typing import Union

import numpy as np
from scipy import ndimage
from skimage.restoration import ellipsoid_kernel
from sklearn.neighbors import NearestNeighbors


def __find_furthest_point_indices(points):
    nbr = NearestNeighbors(n_neighbors=len(points)).fit(points)
    distances, indices = nbr.kneighbors(points)
    ind = np.where(distances == np.max(distances))
    ind2 = [ind[0][0], indices[ind][0]]
    return ind2


def __gaussian_kernel(sigma, shape):
    kernel = np.zeros(shape)
    kernel[tuple(np.int_((np.array(shape) - 1) / 2))] = 255.
    kernel = ndimage.gaussian_filter(kernel, sigma)
    kernel = kernel / np.max(kernel)
    return kernel


def sort_points(points: Union[list, np.ndarray]):
    """
    Sort the coordinates in the order on minimal distance path between them.

    Parameters
    ----------
    points : list
        List of coordinates.

    Returns
    -------
    list:
        Sorted list of coordinates.
    """
    sorted1 = []
    sorted2 = []
    while len(points) >= 2:
        ind = __find_furthest_point_indices(points)
        selected = points[ind]
        if len(sorted1) == 0 or np.linalg.norm(selected[0] - sorted1[-1]) < np.linalg.norm(selected[0] - sorted2[-1]):
            sorted1.append(selected[0])
            sorted2.append(selected[1])
        else:
            sorted1.append(selected[1])
            sorted2.append(selected[0])
        points = points[~np.isin(np.arange(len(points)), ind)]
    if len(points) > 0:
        sorted1 = sorted1 + [points[0]]
    sorted2.reverse()
    sorted1 = sorted1 + sorted2
    points = np.array(sorted1)
    return points


def snap_to_brightest(points: Union[list, np.ndarray], img: np.ndarray, rad: Union[int, list, np.ndarray] = 5,
                      decay_sigma: Union[int, float, list, np.ndarray] = 5.):
    """
    Snap the filament points to the brightest point in the image in the give neighborhood.

    Parameters
    ----------
    points : list
        List of filament coordinates.
    img : np.ndarray
        Image used to find the brightest point.
    rad : int or sequence, optional
        Radius of the neighborhood (in pixels) to consider for identifying the brightest points.
        Can be provided as a list of values for different dimensions.
        Default is 5.
    decay_sigma : scalar or sequence, optional
        Sigma of a Gaussian used to scale image intensities centered on the original annotated point.
        This is done to give preference to the original point, if there are other points with the same intensity.
        Default is 5.
    Returns
    -------

    """
    rad = np.ravel(rad)
    if len(rad) < len(img.shape):
        rad = np.array([rad[0]] * len(img.shape))
    decay_sigma = np.ravel(decay_sigma)
    if len(decay_sigma) < len(img.shape):
        decay_sigma = np.array([decay_sigma[0]] * len(img.shape))
    kernel = (ellipsoid_kernel(rad * 2 + 1, 1) < 1) * 1
    imgpad = np.pad(img, [(r + 1, r + 1) for r in rad])
    points = np.int_(np.round_(points))

    updated_points = []
    for point in points:
        sl = [slice(point[i] + 1, point[i] + 2 * rad[i] + 2) for i in range(len(point))]
        crop = imgpad[tuple(sl)] * kernel * __gaussian_kernel(decay_sigma, kernel.shape)
        shift = np.array(np.where(crop == np.max(crop))).transpose()[0] - rad
        updated_points.append(np.array(point) + shift)
    return np.array(updated_points)
