import numpy as np
from numpy import linalg as LA
import math

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

def A2Euler(A):
  assert (A.shape[0] == A.shape[1]) and np.allclose(A.T @ A, np.eye(A.shape[0])), "A needs to be an orthogonal matrix"
  assert (A.shape[0] == A.shape[1]) and np.round(LA.det(A)) == 1, "Determinant of A must be 1"

  if(A[2][0] < 1):
    if(A[2][0] > -1):
      phi = math.atan2(A[2][1], A[2][2])
      theta = math.asin(-A[2][0])
      psi = math.atan2(A[1][0], A[0][0])
    else:
      phi = 0
      theta = math.pi / 2
      psi = math.atan2(-A[0][1], A[1][1])
  else:
    phi = 0
    theta = - math.pi / 2
    psi = math.atan2(-A[0][1], A[1][1])

  return phi, theta, psi

def AxisAngle2Q(p, phi):
  w = math.cos(phi/2)
  p = normalize(p)

  x = math.sin(phi/2) * p[0]
  y = math.sin(phi/2) * p[1]
  z = math.sin(phi/2) * p[2]

  return np.array([x, y, z, w])

def Q2AxisAngle(q):
  q = normalize(q)

  if(q[3] < 0):
    q = -q

  phi = 2*math.acos(q[3])

  if(abs(q[3]) == 1):
    p = np.array([1, 0, 0])
  else:
    p = normalize(np.array([q[0], q[1], q[2]]))

  return p, phi

if __name__ == "__main__":
  # phi = -math.atan(1/4)
  # theta = -math.asin(8/9)
  # psi = math.atan(4)
  phi = math.pi/2
  theta = 0
  psi = math.pi/2
  print("Pocetni uglovi:")
  print(phi, theta, psi)
  print()

  A = Euler2A(phi, theta, psi)
  print("Euler2A:")
  A = np.array(A, dtype = int)
  print(A)
  print()

  p, phi = AxisAngle(A)
  print("AxisAngle:")
  print(f"osa: {p} i ugao: {phi}")
  print()

  A = Rodrigues(p, phi)
  print("Rodrigues:")
  A = np.array(A, dtype = int)
  print(A)
  print()

  phi, theta, psi = A2Euler(A)
  print(f"A2Euler:")
  print(f"{phi}, {theta}, {psi}")
  print()

  p, phi = AxisAngle(A)
  q = AxisAngle2Q(p, phi)
  print("AxisAngle2Q:")
  print(q)
  print()

  p, phi = Q2AxisAngle(q)
  print("Q2AxisAngle:")
  print(f"osa:{p} i ugao:{phi}")

