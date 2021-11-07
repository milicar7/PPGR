import numpy as np
import numpy.linalg as LA
import math

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

def naivni_algoritam(a, b, c, d, ap, bp, cp, dp):

  d = np.array([d]).T
  dp = np.array([dp]).T

  tmp = np.array([a, b, c]).T
  x = np.matmul(LA.inv(tmp), d)
  p1 = np.array([x[0]*tmp[:, 0], x[1]*tmp[:, 1], x[2]*tmp[:, 2]]).T

  tmp = np.array([ap, bp, cp]).T
  y = np.matmul(LA.inv(tmp), dp)
  p2 = np.array([y[0]*tmp[:, 0], y[1]*tmp[:, 1], y[2]*tmp[:, 2]]).T

  return np.array(np.matmul(p2, LA.inv(p1)))

#print('''
#PRIMERI IZ 1. PDF-A
      #''')

#a = np.array([-3, -1, 1])
#b = np.array([3, -1, 1])
#c = np.array([1, 1, 1])
#d = np.array([-1, 1, 1])
#ap = np.array([-2, -1, 1])
#bp = np.array([2, -1, 1])
#cp = np.array([2, 1, 1])
#dp = np.array([-2, 1, 1])
#print("naivni alg")
#p = naivni_algoritam(a, b, c, d, ap, bp, cp, dp)
#print(p)
#e = np.array([1, 2, 3])
#ep = np.matmul(p, e)
#f = np.array([-8, -2, 1])
#fp = np.matmul(p, f)

#mat = dlt_algoritam([a, b, c, d, e, f, ap, bp, cp, dp, ep, fp])
#mat /= mat[0][0]
#mat *= p[0][0]
#mat = np.array(mat, dtype = int)
#print("dlt algoritam --> dobija se kao kod naivnog")
#print(mat)

#e = afinizuj(e)
#f = afinizuj(f)
#ep = afinizuj(ep)
#fp = afinizuj(fp)

#m = modifikovani_dlt_algoritam([a, b, c, d, e, f, ap, bp, cp, dp, ep, fp])
#m /= m[0][0]
#m *= p[0][0]
#m = np.round(m, 5)
#m = np.array([m], dtype = int)
#print("modif dlt algoritam --> dobija se kao kod naivnog")
#print(m)

#print('''
#PRIMERI IZ 2. PDF-A
      #''')
#a = np.array([1, 1, 1])
#b = np.array([5, 2, 1])
#c = np.array([6, 4, 1])
#d = np.array([-1, 7, 1])

#ap = np.array([0, 0, 1])
#bp = np.array([10, 0, 1])
#cp = np.array([10, 5, 1])
#dp = np.array([0, 5, 1])

#naiv = naivni_algoritam(a, b, c, d, ap, bp, cp, dp)
#print("naivni algoritam - mora 4 tacke:")
#print()
#print(naiv)
#print()

#dlt = dlt_algoritam([a, b, c, d, ap, bp, cp, dp])
#print("dlt algoritam - za 4 tacke:")
#print()
#print(dlt)
#print()

#e = np.array([3, 1, 1])
#ep = np.array([3, -1, 1])
#dlt = dlt_algoritam([a, b, c, d, e, ap, bp, cp, dp, ep])
#print("dlt algoritam - za pet tacaka:")
#print()
#print(dlt)
#print()

#print("dlt nije osetljiv na permutaciju odgovarajucih tacaka:")
#print()
#dlt = dlt_algoritam([a, c, b, d, e, ap, cp, bp, dp, ep])
#print(dlt)
#print()

#print("dlt nije invarijantan na promenu koordinata:")
#print()
#originali_pre = np.array([a, b, c, d, e])
#slike_pre = np.array([ap, bp, cp, dp, ep])

#c1 = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
#c2 = np.array([[1, -1, 5], [1, 1, -2], [0, 0, 1]])
#tacke = np.array([*map(lambda x: c1 @ x.T, originali_pre)])
#slike = np.array([*map(lambda x: c2 @ x.T, slike_pre)])

#dlt_nove = dlt_algoritam(np.concatenate((tacke, slike)))

#dlt_stari =  LA.inv(c2) @ dlt_nove @ c1

#print("dlt pre promene koordinata")
#print()
#print(np.round(dlt, 4))
#print()
#print("dlt posle promene koordinata")
#print()
#print(np.round(dlt_stari * dlt[0][0] / dlt_stari[0][0], 4))
#print()

#print('''
#PRIMERI IZ 2. PDF-A
      #''')

#print("modifikovani DLT algoritam - poredjenje sa DLP algoritmom i invarijantnost u odnosu na tranformaciju koordinata:")
#print()
#dlt = dlt_algoritam([a, c, b, d, e, ap, cp, bp, dp, ep])
#dlt_norm = modifikovani_dlt_algoritam([a, c, b, d, e, ap, cp, bp, dp, ep])
#print("dlt algoritam")
#print()
#print(np.round(dlt, 6))
#print()
#print("modifikovani dlt algoritam")
#print()
#dlt_norm = dlt_norm / dlt_norm[0][0] * dlt[0][0]
#print(np.round(dlt_norm, 6))

#print()
#print("modifikovani dlt ne zavisi od izbora koordinata:")
#print()

#dlt_norm = modifikovani_dlt_algoritam([a, c, b, d, e, ap, cp, bp, dp, ep])

#dlt_norm_nove = modifikovani_dlt_algoritam(np.concatenate((tacke, slike)))

#dlt_norm_stari =  LA.inv(c2) @ dlt_norm_nove @ c1

#print("modifikovani dlt pre promene koordinata")
#print()
#print(np.round(dlt_norm, 4))

#print()

#print("modifikovani dlt posle pormene koordinata")
#print()
#print(np.round(dlt_norm_stari, 4))

#print('''
#PRIMERI IZ 3. PDFA
      #''')

#p = np.array([[0, 3, 5], [4, 0, 0], [-1, -1, 6]])

#a = np.array([-3, 2, 1])
#b = np.array([-2, 5, 2])
#c = np.array([1, 0, 3])
#d = np.array([-7, 3, 1])
#e = np.array([2, 1, 2])
#f = np.array([-1, 2, 1])
#g = np.array([1, 1, 1])

#ap = p @ a
#bp = p @ b
#cp = p @ c
#dp = p @ d
#ep = p @ e
#fp = p @ f
#gp = np.array([8.02, 4, 4])

#print("primenom naivnog alg dobijamo istu matricu", end = '\n')
#matrica_naivnim = naivni_algoritam(a, b, e, d, ap, bp, ep, dp)
#matrica_naivnim = np.round(matrica_naivnim, 0)
#matrica_naivnim = np.array([matrica_naivnim], dtype = int)
#matrica_naivnim = matrica_naivnim[0]
#print(matrica_naivnim, end = '\n')
#print()

#print("primena dlt algoritma", end = '\n')
#matrica_dlt = dlt_algoritam([a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp])
#print()
#print("pre skaliranja")
#print()
#print(matrica_dlt)
#print()
#matrica_dlt = matrica_dlt / matrica_dlt[0][1] * 3

##matrica_dlt = np.round(matrica_dlt, 4)
#print("posle skaliranja")
#print()
#print(matrica_dlt)
#print()

#print("primena modifikovanog dlt algoritma", end = '\n')
#b = afinizuj(b)
#c = afinizuj(c)
#e = afinizuj(e)
#ap = afinizuj(ap)
#bp = afinizuj(bp)
#cp = afinizuj(cp)
#dp = afinizuj(dp)
#ep = afinizuj(ep)
#fp = afinizuj(fp)
#gp = afinizuj(gp)
#matrica_norm = modifikovani_dlt_algoritam([a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp])
#print()
#print("pre skaliranja")
#print(matrica_norm)
#print()
#matrica_norm = matrica_norm / matrica_norm[0][1] * 3
#print("posle skaliranja")
##matrica_norm = np.round(matrica_norm, 4)

#print(matrica_norm)
#print()

#print('''
#PRIMERI IZ 3. PDFA
      #''')

#print("testiramo invarijantnost u odnosu na promenu koordinata")
#cc = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
#originali_pre = [a, b, c, d, e, f, g]
#originali = np.array([*map(lambda x: cc @ x.T, originali_pre)])
#slike_pre = [ap, bp, cp, dp, ep, fp, gp]
#slike = np.array([*map(lambda x: cc @ x.T, slike_pre)])
#print()

#print("PRVO DLT")
#m1 = dlt_algoritam(np.concatenate((originali, slike)))
#m1_stari = LA.inv(cc) @ m1 @ cc
#m1_stari = m1_stari / m1_stari[0][1] * 3

#print("nakon promene:")
##m1_stari= np.round(m1_stari, 4)
#print(m1_stari)
#print()
#print("pre promene")
#print(matrica_dlt)
#print()

#print("SADA MODIF DLT")
#m2 = modifikovani_dlt_algoritam(np.concatenate((originali, slike)))
#m2_stari = LA.inv(cc) @ m2 @ cc
#m2_stari = m2_stari / m2_stari[0][1] * 3

#print("nakon promene:")
##m2_stari= np.round(m2_stari, 4)
#print(m2_stari)
#print()
#print("pre promene")
#print(matrica_norm)
#print()

def unesi_originale():
  print("Unesite originalne tacke: ")
  n = (int)(input("Unesite broj tacaka: "))
  src = []

  for i in range(n):
    print(f"Tacka {i+1}")
    koordinate = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
    src.append(koordinate)
  
  return src

def unesi_slike():
  print("Unesite slike tacaka: ")
  n = (int)(input("Unesite broj tacaka: "))
  dst = []

  for i in range(n):
    print(f"Slika {i+1}")
    koordinate = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
    dst.append(koordinate)
  
  return dst

def unesi_cetiri_tacke():
  print("Unesite 4 tacke za naivni algoritam")

  print("Prvo unesite sve 4 originalne tacke, a zatim 4 slike redom: ")

  print("Prva tacka:")
  a = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Druga tacka:")
  b = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Treca tacka:")
  c = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Cetvrta tacka:")
  d = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])

  print("Prva slika:")
  ap = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Druga silka:")
  bp = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Treca slika:")
  cp = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])
  print("Cetvrta slika:")
  dp = np.array([(float)(input("prva koordinata: ")), (float)(input("druga koordinata: ")), (float)(input("treca koordinata: "))])

  return a, b, c, d, ap, bp, cp, dp


if __name__ == "__main__":
  izbor_algoritma = (int)(input("Izaberite algoritam(0 - naivni algoritam, 1 - dlt algoritam, 2 - modifikovani dlt algoritam): "))
  if izbor_algoritma == 0:
    a, b, c, d, ap, bp, cp, dp = unesi_cetiri_tacke()
    a = afinizuj(a)
    b = afinizuj(b)
    c = afinizuj(c)
    d = afinizuj(d)
    ap = afinizuj(ap)
    bp = afinizuj(bp)
    cp = afinizuj(cp)
    dp = afinizuj(dp)
    print(naivni_algoritam(a, b, c, d, ap, bp, cp, dp))


  elif izbor_algoritma == 1:

    lista = []

    originali = unesi_originale()
    for x in originali:
      x = afinizuj(x)
      lista.append(x)
      
    print("lista")
    print(lista)

    slike = unesi_slike()
    for y in slike:
      y = afinizuj(y)
      lista.append(y)

    print("lista")
    print(lista)

    print(dlt_algoritam(lista))

  elif izbor_algoritma == 2:
    lista = []

    originali = unesi_originale()
    for x in originali:
      x = afinizuj(x)
      lista.append(x)
      
    print("lista")
    print(lista)

    slike = unesi_slike()
    for y in slike:
      y = afinizuj(y)
      lista.append(y)

    print("lista")
    print(lista)

    print(modifikovani_dlt_algoritam(lista))
  else:
    print("Mozete uneti samo 0, 1 ili 2")
