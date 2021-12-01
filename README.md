# filament_annotator_3D

Annotation of filament structures in 3D based on napari viewer

### Annotation instructions:

1. Rotate the image to find a position, where the filament is clearly visible
2. Draw a line over the filament, by holding "Control" and dragging the mouse: this will draw a polygon with potential filament locations
3. Rotate the image to view the filament from another angle and repeat step 2
4. Rotate the image again: this will calculate the filament position from the intersection of the two polygons
5. Repeat steps 1-4 for other filaments

To delete the last added shape (polygon or filament), press "d"