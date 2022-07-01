import itertools

import napari
import numpy as np

from .utils.geom import sort_points, compute_polygon_intersection
from .utils.io import annotation_to_pandas


def annotate_filaments(annotation_layer, output_fn, point_size=1):
    """

    Parameters
    ----------
    annotation_layer : napari layer
        napari shapes layer to add annotations
    output_fn : str
        csv file to save filament coordinates
    point_size : scalar
        Point and line size for display.
        Default is 1.
    Returns
    -------

    """
    near_points = []
    far_points = []
    polygons = []

    @annotation_layer.mouse_drag_callbacks.append
    def draw_polygon_shape(layer, event):
        """
        Draw two polygons in different projections and calculate their intersection
        """
        yield
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

            if len(near_points) > 0 and len(far_points) > 0:
                layer = draw_polygon(layer, near_points, far_points, point_size=point_size)

        yield

    @annotation_layer.mouse_drag_callbacks.append
    def calculate_intersection(layer, event):
        """
        Draw two polygons in different projections and calculate their intersection
        """
        yield
        while event.type == 'mouse_move':
            if 'Control' not in event.modifiers:  # draw a polygon if "Control" is pressed
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
                    mt = compute_polygon_intersection(npt1, npt2, fpt1, fpt2)
                    # add the calculated filament
                    mt = sort_points(mt)  # make sure the filament coordinates are sorted

                    # remove the 2 polygons from the shapes layer
                    layer.selected_data = set(range(layer.nshapes - 2, layer.nshapes))
                    layer.remove_selected()

                    # add the calculated filament
                    layer.add(mt, shape_type='path', edge_color='green', edge_width=point_size)

                    # clear the polygons array
                    polygons[0] = None
                    polygons[1] = None
                    polygons.clear()

                    # save current annotations to a csv file
                    annotation_to_pandas(layer.data[1:]).to_csv(output_fn, index=False)

            yield

    @annotation_layer.bind_key('p')
    def delete_the_last_shape(layer, show_message=True):
        """
        Remove the last added shape (polygon or filament)

        """
        if layer.nshapes > 1:
            msg = 'delete the last added shape'
            layer.selected_data = set(range(layer.nshapes - 1, layer.nshapes))
            if len(polygons) > 0:
                polygons.pop()
            layer.remove_selected()
        else:
            msg = 'no shapes to delete'
        if show_message:
            layer.status = msg
            print(msg)
        annotation_to_pandas(layer.data[1:]).to_csv(output_fn, index=False)
        near_points.clear()
        far_points.clear()

    @annotation_layer.bind_key('d')
    def delete_the_last_point(layer):
        """
        Remove the last added point

        """
        if len(near_points) > 0 and len(far_points) > 0:
            near_points.pop()
            far_points.pop()
            if len(near_points) > 0:
                layer = draw_polygon(layer, near_points.copy(), far_points.copy())
            else:
                delete_the_last_shape(layer, show_message=False)


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


def draw_polygon(layer, near_points: list, far_points: list, color: str = 'red', point_size=1):
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
    point_size : scalar
        Point and line size for display.
        Default is 1.

    Returns
    -------
    Updated shapes layer
    """
    if len(near_points) < 2:
        near_points.append(np.array(near_points[0]) + 1)
        far_points.append(np.array(far_points[0]) + 1)
    far_points_reverse = far_points.copy()
    far_points_reverse.reverse()
    polygon = np.array(near_points + far_points_reverse)
    if np.array((layer.data[-1][0] == polygon[0])).all():
        layer.selected_data = set(range(layer.nshapes - 1, layer.nshapes))
        layer.remove_selected()
    layer.add(
        polygon,
        shape_type='polygon',
        edge_width=point_size,
        edge_color=color
    )
    return layer
