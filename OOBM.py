import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt



no_of_points = 30
rng = np.random.default_rng()
points = rng.random((no_of_points, 2))

hull = ConvexHull(points)
print(hull.simplices)
hull_points = points[ConvexHull(points).vertices]
print(hull_points)


def OOBM(points):
    pi2 = np.pi/2.
    # calculate edge angles
    # make an array of zeros of the same shape and size
    edge = np.zeros((len(hull_points)-1, 2))
    edge = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edge)))
    # The inverse of tan, so that if y = tan(x) then x = arctan(y).
    angles = np.arctan2(edge[:, 1], edge[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([  # Stack arrays in sequence vertically
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    print(rotations)
    rotations = rotations.reshape((-1, 2, 2))

    # rotate the hull
    rotationalPoints = np.dot(rotations, hull_points.T)

    # find the extreme points
    #nanmin nanmax Return minimum of an array or minimum along an axis
    minx = np.nanmin(rotationalPoints[:, 0], axis=1)
    maxx = np.nanmax(rotationalPoints[:, 0], axis=1)
    miny = np.nanmin(rotationalPoints[:, 1], axis=1)
    maxy = np.nanmax(rotationalPoints[:, 1], axis=1)


    areas = (maxx - minx) * (maxy - miny)
    minBox = np.argmin(areas)

    # return the best box
    x1 = maxx[minBox]
    x2 = minx[minBox]
    y1 = maxy[minBox]
    y2 = miny[minBox]
    r = rotations[minBox]

    rect = np.zeros((4, 2))
    rect[0] = np.dot([x1, y2], r)
    rect[1] = np.dot([x2, y2], r)
    rect[2] = np.dot([x2, y1], r)
    rect[3] = np.dot([x1, y1], r)
    return rect


for n in range(10):
    minBox = OOBM(points)
    plt.fill(minBox[:, 0], minBox[:, 1])
    plt.axis('equal')


plt.plot(points[:, 0], points[:, 1], 'o')
# (ndarray of ints, shape (nfacet, ndim)) Indices of points forming the simplical facets of the convex hull.
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


plt.show()
