#potrebno je imati ovu verziju zbog sift-a
#!pip install opencv-contrib-python==4.4.0.44

import numpy as np
import numpy.linalg as LA
import math

import cv2
#na google colabu nije bilo moguce koristiti cv2.imshow
from google.colab.patches import cv2_imshow

def afinizuj(tacka):
  x = tacka[0]/tacka[2]
  y = tacka[1]/tacka[2]
  
  return np.array([x, y, 1])

def izracunaj_afino_teziste(tacke):
  ukupno = len(tacke)
  c_x = 0
  c_y = 0

  for tacka in tacke:
    c_x += tacka[0]
    c_y += tacka[1]

  return c_x/ukupno, c_y/ukupno

def prosek_rastojanja(tacke):

  ukupno = len(tacke)
  rastojanje = 0

  for tacka in tacke:
    rastojanje += math.sqrt(tacka[0]**2 + tacka[1]**2)
  
  return rastojanje/ukupno

def normalizuj(tacke):
  
  (c_x, c_y) = izracunaj_afino_teziste(tacke)

  translirane_tacke = []
  for tacka in tacke:
    translirane_tacke.append([tacka[0] - c_x, tacka[1] - c_y])

  prosek = prosek_rastojanja(translirane_tacke)
  koeficijent = math.sqrt(2) / prosek

  normalizovane_tacke = []
  for tacka in translirane_tacke:
    normalizovane_tacke.append([tacka[0]*koeficijent, tacka[1]*koeficijent, 1])

  G = np.array([[1, 0, -c_x], [0, 1, -c_y], [0, 0, 1]])
  S = np.array([[koeficijent, 0, 0], [0, koeficijent, 0], [0, 0, 1]])
  T = np.matmul(S, G)

  return normalizovane_tacke, T

def dlt_algoritam(args):

  n = len(args)
  n = n//2

  A = np.array([]).reshape(0, 9)

  for i in range(0, n):

    x = args[i]
    xp = args[i + n]

    prvi = np.array([[0, 0, 0, -xp[2]*x[0], -xp[2]*x[1], -xp[2]*x[2], xp[1]*x[0], xp[1]*x[1], xp[1]*x[2]]])
    drugi = np.array([[xp[2]*x[0], xp[2]*x[1], xp[2]*x[2], 0, 0, 0, -xp[0]*x[0], -xp[0]*x[1], -xp[0]*x[2]]])

    tmp = np.concatenate((prvi, drugi))

    A = np.concatenate((A, tmp))
  _, _, v = LA.svd(A)
  
  p = v[-1]
  
  return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]])

def modifikovani_dlt_algoritam(args):

  n = len(args)
  n //=2
  norm_orig, t = normalizuj(args[:n])
  norm_slike,tp = normalizuj(args[n:])
  
  pn = dlt_algoritam(norm_orig + norm_slike)

  p = LA.inv(tp) @ pn @ t

  return p


def otkloni_proj_distorziju(koordinate_originala, koordinate_pravougaonika):

  original = cv2.imread('building.jpg')

  #koordinate_originala = np.array([[474, 742, 1], [471, 136, 1], [1450, 525, 1], [1449, 846, 1]])
  #koordinate_pravougaonika = [[474, 742, 1], [474, 136, 1], [1449, 136, 1], [1449, 742, 1]]

  dlt_matrica = dlt_algoritam(np.concatenate((koordinate_originala, koordinate_pravougaonika)))

  bez_distorzije = cv2.warpPerspective(original, dlt_matrica, (original.shape[1], original.shape[0]))

  cv2_imshow(bez_distorzije)

def unesi_piksele():
  x = []

  for i in range(4):
    print(f"Tacka {i+1}")
    koordinate = np.array([(int)(input("prva koordinata: ")), (int)(input("druga koordinata: ")), 1])
    x.append(koordinate)
  
  src = np.array([np.array(xi) for xi in x])
  print(src)
  return src

def panorama():
  slika_ = cv2.imread('slika1.jpg')
  slika1 = cv2.cvtColor(slika_,cv2.COLOR_BGR2GRAY)

  slika = cv2.imread('slika2.jpg')
  slika2 = cv2.cvtColor(slika,cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create()

  keypoint1, descriptor1 = sift.detectAndCompute(slika1,None)
  keypoint2, descriptor2 = sift.detectAndCompute(slika2,None)

  match = cv2.BFMatcher()
  matches = match.knnMatch(descriptor1,descriptor2,k=2)

  filter = []
  for m,n in matches:
      if m.distance < 0.4*n.distance:
          filter.append(m)


  parametri = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    flags = 2)

  slika3 = cv2.drawMatches(slika_,keypoint1,slika,keypoint2, filter,None,**parametri)
  print("Iscrtavaju se zraci koji povezuju podudarene tacke...")
  cv2_imshow(slika3)

  minimum_podudaranja = 10
  if len(filter) > minimum_podudaranja:
    
      originali = np.float32([ keypoint1[m.queryIdx].pt for m in filter]).reshape(-1,1,2)
      slike = np.float32([ keypoint2[m.trainIdx].pt for m in filter ]).reshape(-1,1,2)

      lista = []
      n = len(originali)
      for i in range(n):
        lista.append(originali[i][0])

      for i in range(n):
        lista.append(slike[i][0])
            
      matrica = modifikovani_dlt_algoritam(lista)

      #matrica, _ = cv2.findHomography(originali, slike, cv2.RANSAC, 5.0)

      h,w = slika1.shape
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      dst = cv2.perspectiveTransform(pts, matrica)
  else:
      print("Nedovoljno podudaranja- %d/%d", (len(filter)/minimum_podudaranja))

  dst = cv2.warpPerspective(slika_,matrica,(slika.shape[1] + slika_.shape[1], slika.shape[0]))
  dst[0:slika.shape[0],0:slika.shape[1]] = slika
  print("Iscrtava se panorama: ")
  cv2_imshow(dst)

if __name__ == "__main__":
  print("***************************************")
  print("PRIMENE")
  print("***************************************")

  print("OTKLANJANJE PROJEKTIVNE DISTORZIJE")

  print("Pod komentarom cu ostaviti koordinate koje sam ja unosila za test sliku koju sam poslala u okviru domaceg zadatka")
  #koordinate_originala = np.array([[474, 742, 1], [471, 136, 1], [1450, 525, 1], [1449, 846, 1]])
  #koordinate_pravougaonika = [[474, 742, 1], [474, 136, 1], [1449, 136, 1], [1449, 742, 1]]
  print("Unesite koordinate 4 tacke u pikselima:")
  pikseli = unesi_piksele()
  print("Unesite koordinate pravougaonika u pikselima:")
  pravougaonik = unesi_piksele()
  print("Otklanja se projektivna distorzija...")
  otkloni_proj_distorziju(pikseli, pravougaonik)

  print("PANORAMA")
  
  panorama()

