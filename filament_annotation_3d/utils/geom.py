from typing import Union

import numpy as np
from Geometry3D import *
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


def tetragon_intersection(p1: list, p2: list):
    """
    Calculate intersection of two tetragons in 3D

    Parameters
    ----------
    p1, p2 : list
        List of tetragon coordinates

    Returns
    -------
    list or None:
        List of (two) coordinate of the intersection line or None if no intersection exists.
    """
    t = []
    p1 = np.array(p1)
    p2 = np.array(p2)
    if p1.shape[1] > 3:
        t = list(p1[0][:-3])
        p1 = p1[:, -3:]
        p2 = p2[:, -3:]
    p1 = list(set([Point(*coords) for coords in p1]))
    p2 = list(set([Point(*coords) for coords in p2]))

    if len(p1) > 2 and len(p2) > 2:
        plane1 = ConvexPolygon(p1)
        plane2 = ConvexPolygon(p2)
        inter = intersection(plane1, plane2)
        if inter is not None:
            inter = [t + list(pt) for pt in inter]
        return inter
    else:
        return None


def smooth_points(points, sig, maxpoints):
    points = np.array(points)
    points = ndimage.gaussian_filter(points, [sig, 0])
    if maxpoints is not None:
        ind = np.int_(np.linspace(0, len(points) - 1, maxpoints, endpoint=False))
        points = np.array(points)[ind]
    return points


def compute_polygon_intersection(npt1: np.ndarray, npt2: np.ndarray,
                                 fpt1: np.ndarray, fpt2: np.ndarray,
                                 sigma=1, maxpoints=None):
    """
    Calculate intersection of two non-convex polygons represented by a list of near and far points.

    Parameters
    ----------
    npt1 : np.ndarray
        Near points of the first polygon.
    npt2 : np.ndarray
        Near points of the second polygon.
    fpt1 : np.ndarray
        Far points of the first polygon.
    fpt2 : np.ndarray
        Far points of the second polygon.
    sigma : float
        Gaussian filter size in pixels to smooth the polygon points array before computing intersection.
        Default: 1
    maxpoints : int, optional
        If provided, the number of points will be reduced to this number before computing intersection.
        Default: None
    Returns
    -------
    np.ndarray:
        n x d array of the intersections coordinates,
        where n is the number of points, d is the number of dimensions.
    """
    mt = []
    npt1 = smooth_points(npt1, sigma, maxpoints)
    npt2 = smooth_points(npt2, sigma, maxpoints)
    fpt1 = smooth_points(fpt1, sigma, maxpoints)
    fpt2 = smooth_points(fpt2, sigma, maxpoints)

    for i in range(len(npt1) - 1):
        for j in range(len(npt2) - 1):
            p1 = [npt1[i], npt1[i + 1], fpt1[i + 1], fpt1[i]]
            p2 = [npt2[j], npt2[j + 1], fpt2[j + 1], fpt2[j]]
            inter = tetragon_intersection(p1, p2)
            if inter is not None:
                if len(mt) == 0:
                    mt = inter
                else:
                    mt = np.concatenate([mt, inter], axis=0)

    mt = np.array(list(set([tuple(np.round(mt[i], 1)) for i in range(len(mt))])))
    return mt


def __find_furthest_point_indices(points):
    nbr = NearestNeighbors(n_neighbors=len(points)).fit(points)
    distances, indices = nbr.kneighbors(points)
    ind = np.where(distances == np.max(distances))
    ind2 = [ind[0][0], indices[ind][0]]
    return ind2


def sort_points(points: Union[list, np.ndarray]):
    """    Sort the coordinates in the order on minimal distance path between them.

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
