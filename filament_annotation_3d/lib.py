import itertools
from typing import Union

import napari
import numpy as np
import pandas as pd
from Geometry3D import *
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


def annotate_filaments(annotation_layer, output_fn, maxpoints: int = 10):
    """

    Parameters
    ----------
    annotation_layer : napari layer
        napari shapes layer to add annotations
    output_fn : str
        csv file to save filament coordinates
    maxpoints : int, optional
        Maximum points in the annotated filament.
        Used for regularization.
        Default: 10
    Returns
    -------

    """
    near_points = []
    far_points = []
    polygons = []

    @annotation_layer.mouse_drag_callbacks.append
    def draw_polygons(layer, event):
        """
        Draw two polygons in different projections and calculate their intersection
        """
        yield
        while event.type == 'mouse_move':
            if 'Control' in event.modifiers:  # draw a polygon if "Control" is pressed
                # get the near a far points of the mouse position
                near_point, far_point = layer.get_ray_intersections(
                    event.position,
                    event.view_direction,
                    event.dims_displayed
                )
                # append to the array of near and far points
                if (near_point is not None) and (far_point is not None):
                    near_points.append(near_point)
                    far_points.append(far_point)

                # draw a polygon from the array of near and far points if there are > 3 of them
                if len(near_points) > 3:
                    layer = draw_polygon(layer, near_points, far_points)

            else:  # normal rotation when "Control" is not pressed
                # add a polygon if there are some point saved
                if len(near_points) > 0:
                    polygons.append([near_points.copy(), far_points.copy()])

                # clear the points array
                near_points.clear()
                far_points.clear()

                # if there are 2 or more polygons, calculate their intersection
                if len(polygons) >= 2:
                    npt1 = polygons[0][0]
                    npt2 = polygons[1][0]
                    fpt1 = polygons[0][1]
                    fpt2 = polygons[1][1]
                    mt = compute_polygon_intersection(npt1, npt2, fpt1, fpt2, maxpoints=maxpoints)
                    # add the calculated filament
                    mt = sort_points(mt)  # make sure the filament coordinates are sorted

                    # remove the 2 polygons from the shapes layer
                    layer.selected_data = set(range(layer.nshapes - 2, layer.nshapes))
                    layer.remove_selected()

                    # add the calculated filament
                    layer.add(mt, shape_type='path', edge_color='green', edge_width=1)

                    # clear the polygons array
                    polygons[0] = None
                    polygons[1] = None
                    polygons.clear()

                    # save current annotations to a csv file
                    annotation_to_pandas(layer.data[1:]).to_csv(output_fn, index=False)

            yield

    @annotation_layer.bind_key('d')
    def delete_the_last_shape(layer):
        """
        Remove the last added shape (polygon or filament)

        """
        if layer.nshapes > 1:
            msg = 'delete the last added shape'
            layer.selected_data = set(range(layer.nshapes - 1, layer.nshapes))
            if len(polygons) > 0:
                _ = polygons.pop()
            layer.remove_selected()
        else:
            msg = 'no shapes to delete'
        layer.status = msg
        print(msg)
        annotation_to_pandas(layer.data[1:]).to_csv(output_fn, index=False)


def create_random_lines(size: int, n: int = 10, sigma: float = 0.8) -> np.ndarray:
    """
    Generate an image (cube) with random lines.

    Parameters
    ----------
    size : int
        Size of the cube side in pixels
    n : int, optional
        Number of random lines.
        Default: 10
    sigma : float, optional
        Size of the Gaussian filter to smooth the line image.
        Default: 0.8
    Returns
    -------
    np.ndarray:
        Image cube with random lines
    """
    img = np.zeros([size, size, size])
    for i in range(n):
        start, end = np.random.randint(0, size, (2, 3))
        ind = np.int_(np.linspace(start, end, 100, endpoint=False))
        img[tuple(ind.transpose())] = 255
    img = ndimage.gaussian_filter(img, sigma)
    return img


def add_annotation_layer(viewer: napari.Viewer):
    """
    Add an annotation layer to napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        napari viewer
    Returns
    -------
    napari shapes layer with a bounding box shape
    """
    shape = viewer.layers[0].data.shape

    # add a bounding box to set the coordinates range
    bbox = list(itertools.product(*[np.arange(2)
                                    for i in range(len(shape[-3:]))]))
    if len(shape) > 3:
        bbox = [(0,) + b for b in bbox]
    bbox = bbox * np.array(shape)

    layer = viewer.add_shapes(bbox,
                              name='annotations',
                              shape_type='path',
                              edge_width=0)
    return layer


def draw_polygon(layer, near_points: list, far_points: list, color: str = 'red'):
    """
    Draw a polygon between provided near and far points.

    Parameters
    ----------
    layer : napari shapes layer
        napari shapes layer with annotations
    near_points : list
        List of polygon coordinates nearer to the viewer.
    far_points : list
        List of polygon coordinates further from the viewer.
    color : str, optional
        Color of the polygon

    Returns
    -------
    Updated shapes layer
    """
    far_points_reverse = far_points.copy()
    far_points_reverse.reverse()
    polygon = np.array(near_points + far_points_reverse)
    if np.array((layer.data[-1][0] == polygon[0])).all():
        layer.selected_data = set(range(layer.nshapes - 1, layer.nshapes))
        layer.remove_selected()
    layer.add(
        polygon,
        shape_type='polygon',
        edge_width=1,
        edge_color=color
    )
    return layer


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


def annotation_to_pandas(data: list) -> pd.DataFrame:
    """
    Convert list of path to a pandas table with coordinates.

    Parameters
    ----------
    data : list
        List of paths, each of which is a list of coordinates.

    Returns
    -------
    pd.DataFrame:
        pandas DataFrame with coordinates
    """

    df = pd.DataFrame()
    if len(data) > 0:
        columns = ['t', 'z', 'y', 'x']
        columns = columns[-data[0].shape[1]:]
        for i, d in enumerate(data):
            cur_df = pd.DataFrame(d, columns=columns)
            cur_df['id'] = i
            df = pd.concat([df, cur_df], ignore_index=True)
    return df
