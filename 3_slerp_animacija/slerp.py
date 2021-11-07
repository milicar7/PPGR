import numpy as np
from numpy import linalg as LA
import math

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits import mplot3d

#needed for google colab
rc('animation', html='jshtml')

def norm(p):
  n = p.size
  sum = 0
  for i in range(n):
    sum += p[i]**2
  return math.sqrt(sum)

def normalize(p):
  n = norm(p)
  return p/n

def mul(p):
  pt = p
  arr1 = np.array([*map(lambda x: x * p[0], pt)])
  arr2 = np.array([*map(lambda x: x * p[1], pt)])
  arr3 = np.array([*map(lambda x: x * p[2], pt)])

  return np.array([arr1, arr2, arr3])

def Rodrigues(p, phi):
  p = normalize(p)
  ppt = mul(p)
  E = np.eye(3)
  px = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
  A = ppt + math.cos(phi) * (E - ppt) + math.sin(phi) * px
  return A

def Euler2A(phi, theta, psi):
  rot_x = Rodrigues(np.array([1, 0, 0]), phi)
  rot_y = Rodrigues(np.array([0, 1, 0]), theta)
  rot_z = Rodrigues(np.array([0, 0, 1]), psi)

  return rot_z @ rot_y @ rot_x

def AxisAngle(A):
  assert (A.shape[0] == A.shape[1]) and np.allclose(A.T @ A, np.eye(A.shape[0])), "A needs to be an orthogonal matrix"
  assert (A.shape[0] == A.shape[1]) and np.round(LA.det(A)) == 1, "Determinant of A must be 1"

  original = A
  E = np.eye(3)
  A = A-E

  p = np.cross(A[0], A[1])
  u = A[0]
  up = original @ u

  phi = math.acos(np.dot(u, up) / (norm(u) * norm(up)))

  if(LA.det(np.array([u, up, p])) < 0):
    p = -p
    phi = 2*math.pi - phi

  return normalize(p), phi

def AxisAngle2Q(p, phi):
  w = math.cos(phi/2)
  p = normalize(p)

  x = math.sin(phi/2) * p[0]
  y = math.sin(phi/2) * p[1]
  z = math.sin(phi/2) * p[2]

  return np.array([x, y, z, w])

def Euler2Q(phi, theta, psi):
    A = Euler2A(phi, theta, psi)
    p, phi = AxisAngle(A)
    q = AxisAngle2Q(p, phi)

    return q

def lerp(q1, q2, tm, t):
    q = (1-(t/tm))*q1 + (t/tm)*q2
    return q

def slerp(q1, q2, tm, t):
    q1 = normalize(q1)
    q2 = normalize(q2)

    cos0 = np.dot(q1, q2)

    if cos0 < 0:
        q1 = -1 * q1
        cos0 = -cos0
    if cos0 > 0.95:
        return lerp(q1, q2, tm, t)

    phi0 = math.acos(cos0)

    q = (math.sin(phi0*(1-t/tm))/math.sin(phi0))*q1 + (math.sin(phi0*(t/tm))/math.sin(phi0))*q2

    return q

def mkQuaternion(v, w):
    return np.array([v[0], v[1], v[2], w])

def qInv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]]) / (norm(q) ** 2)

def qmul(q1, q2):
    v = q1[0:3]
    w = q1[3]

    v1 = q2[0:3]
    w1 = q2[3]

    return mkQuaternion(np.cross(v, v1) + w*v1 + w1*v, w*w1 - np.dot(v, v1))

def transform(p, q):
    qi = qInv(q)
    p = np.append([p], 0)

    return qmul(qmul(q, p), qi)[:-1]

if __name__ == "__main__":
    tm =100

    p1 = np.array([0, -5, 2])
    q1 = Euler2Q(0, 0, 5*math.pi/6)

    p2 = np.array([3, 4, 0])
    q2 = Euler2Q(math.pi / 4, - math.pi / 4, 3*math.pi / 4)

    #places a figure in the canvas that is exactly as large the canvas itself
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1], projection = '3d')
    #ax.axis('off')

    ax.set_xlim((-6, 6))
    ax.set_xlabel('X')

    ax.set_ylim((-6, 6))
    ax.set_ylabel('Y')

    ax.set_zlim((-6, 6))
    ax.set_zlabel('Z')

    colors = ['r', 'g', 'b']

    start_pts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    end_pts = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])


    for i in range(3):
        #drawing the start position
        start = transform(start_pts[i], q1)
        end = transform(end_pts[i], q1)
        start += p1
        end += p1
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], colors[i])

        #drawing the end positon
        start = transform(start_pts[i], q2)
        end = transform(end_pts[i], q2)
        start += p2
        end += p2
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], colors[i])

    current = np.array(sum([ax.plot([], [], [], c) for c in colors], []))

    def animate(frame):
        p = lerp(p1, p2, tm, frame)
        q = slerp(q1, q2, tm, frame)

        for c, start, end in zip(current, start_pts, end_pts):
            start = transform(start, q)
            end = transform(end, q)
            start += p
            end += p

            c.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
            c.set_3d_properties(np.array([start[2], end[2]]))

        fig.canvas.draw()


    anim = animation.FuncAnimation(fig, animate, frames=tm, interval=20, repeat=True, repeat_delay=300)

    plt.show()

anim.save('animation.gif', writer = "pillow", fps=10 )

