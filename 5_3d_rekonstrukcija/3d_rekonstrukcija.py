import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# np.set_printoptions(suppress=True)

x1 = np.array([814, 111, 1.])
y1 = np.array([913, 445, 1.])

x2 = np.array([951, 160, 1.])
y2 = np.array([813, 556, 1.])

x3 = np.array([989, 122, 1.])
y3 = np.array([920, 612, 1.])

x4 = np.array([855, 79, 1.])
y4 = np.array([1015, 490, 1.])

x5 = np.array([791, 304, 1.])

x6 = np.array([913, 357, 1.])
y6 = np.array([776, 770, 1.])

x7 = np.array([950, 318, 1.])
y7 = np.array([864, 822, 1.])

y8 = np.array([955, 700, 1.])

x9 = np.array([323, 344, 1.])
y9 = np.array([298, 73, 1.])

x10 = np.array([454, 368, 1.])
y10 = np.array([251, 119, 1.])

x11 = np.array([508, 270, 1.])
y11 = np.array([370, 137, 1.])

x12 = np.array([386, 249, 1.])
y12 = np.array([414, 90, 1.])

x13 = np.array([363, 558, 1.])

x14 = np.array([477, 583, 1.])
y14 = np.array([288, 326, 1.])

x15 = np.array([526, 486, 1.])
y15 = np.array([395, 342, 1.])

y16 = np.array([434, 287, 1.])

x17 = np.array([137, 550, 1.])

x18 = np.array([438, 756, 1.])
y18 = np.array([136, 320, 1.])

x19 = np.array([816, 383, 1.])
y19 = np.array([525, 527, 1.])

x20 = np.array([546, 253, 1.])
y20 = np.array([745, 345, 1.])

x21 = np.array([174, 653, 1.])

x22 = np.array([450, 861, 1.])
y22 = np.array([162, 425, 1.])

x23 = np.array([807, 489, 1.])
y23 = np.array([535, 640, 1.])

y24 = np.array([736, 451, 1.])


def find_with_cross(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10):
    return np.cross(
         np.cross(np.cross(np.cross(n1, n2), np.cross(n3, n4)), n5),
         np.cross(np.cross(np.cross(n6, n7), np.cross(n8, n9)), n10)
    )


def px_coords(x):
    return np.round(x / x[-1])


x8 = find_with_cross(x1, x5, x2, x6, x4, x1, x4, x2, x3, x5)
x8 = px_coords(x8)
# print(x8)

x16 = find_with_cross(x9, x13, x10, x14, x12, x9, x12, x10, x11, x13)
x16 = px_coords(x16)
# print(x16)

x24 = find_with_cross(x17, x21, x18, x22, x20, x17, x20, x18, x19, x21)
x24 = px_coords(x24)
# print(x24)

y5 = find_with_cross(y2, y6, y3, y7, y1, y2, y1, y3, y4, y6)
y5 = px_coords(y5)
# print(y5)

y13 = find_with_cross(y10, y14, y11, y15, y9, y10, y9, y11, y12, y14)
y13 = px_coords(y13)
# print(y13)

y17 = find_with_cross(y18, y19, y22, y23, y20, y19, y20, y23, y24, y18)
y17 = px_coords(y17)
# print(y17)

y21 = find_with_cross(y18, y22, y19, y23, y17, y18, y17, y19, y20, y22)
y21 = px_coords(y21)
# print(y21)

# xx = np.array([x1, x2, x3, x4, x6, x7, x9, x10, x11, x12, x14, x15, x18, x19, x20, x22, x23])
# yy = np.array([y1, y2, y3, y4, y6, y7, y9, y10, y11, y12, y14, y15, y18, y19, y20, y22, y23])
xx = np.array([x1, x2, x3, x4, x9, x10, x11, x12])
yy = np.array([y1, y2, y3, y4, y9, y10, y11, y12])


def eq(x, y):
    return np.array([x[0] * y[0], x[1] * y[0], x[2] * y[0],
                     x[0] * y[1], x[1] * y[1], x[2] * y[1],
                     x[0] * y[2], x[1] * y[2], x[2] * y[2]])


def eq_matrix():
    return np.array([eq(x, y) for x, y in zip(xx, yy)])


_, _, v = LA.svd(eq_matrix())
ff = v[-1].reshape(3, 3)

u, d, v = LA.svd(ff)
e1 = v[-1]
e1 = (1 / e1[2]) * e1

e2 = u.T[-1]
e2 = (1 / e2[2]) * e2

d1 = np.diag([1, 1, 0]) @ d
d1 = np.diag(d1)
ff1 = u @ d1 @ v

t1 = np.hstack([np.eye(3), np.zeros(3).reshape(3, 1)])


def vec(vect):
    return np.array([[0, -vect[2], vect[1]],
                     [vect[2], 0, -vect[0]],
                     [-vect[1], vect[0], 0]])


t2 = np.hstack([vec(e2) @ ff1, e2.reshape(3, 1)])


def triangulation_eq(x, y):
    return np.array([x[1] * t1[2] - x[2] * t1[1], -x[0] * t1[2] + x[2] * t1[0],
                     y[1] * t2[2] - y[2] * t2[1], -y[0] * t2[2] + y[2] * t2[0]])


def affinize(vect):
    return (vect / vect[-1])[:-1]


def coords_3d(x, y):
    return affinize(LA.svd(triangulation_eq(x, y))[-1][-1])


def times_400(x):
    return np.diag([1.0, 1.0, 400]) @ x


im1 = np.array([x1, x2, x3, x4, x5, x6, x7, x8,
                x9, x10, x11, x12, x13, x14, x15, x16,
                x17, x18, x19, x20, x21, x22, x23, x24])
im2 = np.array([y1, y2, y3, y4, y5, y6, y7, y8,
                y9, y10, y11, y12, y13, y14, y15, y16,
                y17, y18, y19, y20, y21, y22, y23, y24])

reconstructed = [coords_3d(x, y) for x, y in zip(im1, im2)]
reconstructed400 = [times_400(x) for x in reconstructed]
# print(reconstructed400)

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")


def plot(i, j, clr):
    p1 = reconstructed400[i-1]
    p2 = reconstructed400[j-1]
    ax.plot3D(xs=[p1[0], p2[0]], ys=[p1[1], p2[1]], zs=[p1[2], p2[2]], color=clr)


color = 'yellow'
plot(1, 2, color)
plot(2, 3, color)
plot(3, 4, color)
plot(4, 1, color)
plot(5, 6, color)
plot(6, 7, color)
plot(7, 8, color)
plot(8, 5, color)
plot(1, 5, color)
plot(2, 6, color)
plot(3, 7, color)
plot(4, 8, color)

color = 'yellowgreen'
plot(9, 10,  color)
plot(10, 11, color)
plot(11, 12, color)
plot(12, 9,  color)
plot(13, 14, color)
plot(14, 15, color)
plot(15, 16, color)
plot(16, 13, color)
plot(9, 13,  color)
plot(10, 14, color)
plot(11, 15, color)
plot(12, 16, color)

color = 'black'
plot(17, 18, color)
plot(18, 19, color)
plot(19, 20, color)
plot(20, 17, color)
plot(21, 22, color)
plot(22, 23, color)
plot(23, 24, color)
plot(24, 21, color)
plot(17, 21, color)
plot(18, 22, color)
plot(19, 23, color)
plot(20, 24, color)

plt.show()
