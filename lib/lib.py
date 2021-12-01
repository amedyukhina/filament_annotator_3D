import numpy as np
from scipy import ndimage
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import itertools
from Geometry3D import *

def create_random_lines(size, n=10, sigma=0.8):
    img = np.zeros([size, size, size])
    for i in range(n):
        start, end = np.random.randint(0, size, (2, 3))
        ind = np.int_(np.linspace(start, end, 100, endpoint=False))
        img[tuple(ind.transpose())] = 255
    img = ndimage.gaussian_filter(img, sigma)
    return img


def add_annotation_layer(viewer):
    shape = viewer.layers[0].data.shape
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


def draw_polygon(layer, near_points, far_points, remove_previous=True, color='red'):
    far_points_reverse = far_points.copy()
    far_points_reverse.reverse()
    polygon = np.array(near_points + far_points_reverse)
    if remove_previous and (layer.data[-1][0] == polygon[0]).all():
        layer.selected_data = set([layer.nshapes - 1])
        layer.remove_selected()
    layer.add(
        polygon,
        shape_type='polygon',
        edge_width=1,
        edge_color=color
    )
    return layer


def tetragon_intersection(p1, p2):
    p1 = list(set([Point(*coords) for coords in p1]))
    p2 = list(set([Point(*coords) for coords in p2]))
    if len(p1) > 2 and len(p2) > 2:
        plane1 = ConvexPolygon(p1)
        plane2 = ConvexPolygon(p2)
        inter = intersection(plane1,plane2)
        if inter is not None:
            inter = [list(pt) for pt in inter]
        return inter
    else:
        return None
    
def compute_polygon_intersection(npt1, npt2, fpt1, fpt2):
    mt = []
    for i in range(len(npt1)-1):
        for j in range(len(npt2)-1):
            p1 = [npt1[i], npt1[i+1], fpt1[i+1], fpt1[i]]
            p2 = [npt2[j], npt2[j+1], fpt2[j+1], fpt2[j]]
            inter = tetragon_intersection(p1, p2)
            if inter is not None:
                if len(mt) == 0:
                    mt = inter
                else:
                    mt = np.concatenate([mt, inter], axis=0)
    return mt


def sort_points(points, n_neighbors=10):
    clf = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    mindist = np.inf
    minidx = 0
    
    for i in range(len(points)):
        p = paths[i]           # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i

    order = paths[minidx]
    return points[order]


def annotation_to_pandas(data):
    df = pd.DataFrame()
    for i, d in enumerate(data):
        cur_df = pd.DataFrame(d, columns=['z', 'y', 'x'])
        cur_df['id'] = i
        df = pd.concat([df, cur_df], ignore_index=True)
    return df