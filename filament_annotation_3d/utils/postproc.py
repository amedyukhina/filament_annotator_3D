from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


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
