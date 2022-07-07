import numpy as np
from scipy import ndimage


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
