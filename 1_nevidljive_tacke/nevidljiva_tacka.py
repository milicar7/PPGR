import numpy as np

def affinize(coordinates):
    # c = xb1[2]
    # xb1 = np.round(xb1/xb1[2])
    # xb1 = np.array(xb1, dtype=int)

    coordinates = np.round(coordinates/coordinates[2])
    coordinates = np.array(coordinates, dtype=int)
    return coordinates


def nevidljiva(p1, p2, p3, p5, p6, p7, p8):

    #prevodimo u homogene koordinate
    p1 = np.append(p1, 1)
    p2 = np.append(p2, 1)
    p3 = np.append(p3, 1)
    p5 = np.append(p5, 1)
    p6 = np.append(p6, 1)
    p7 = np.append(p7, 1)
    p8 = np.append(p8, 1)

    #racunamo tacku xb kao teziste tri tacke
    xb1 = np.cross(np.cross(p2, p6), np.cross(p1, p5))
    xb1 = affinize(xb1)
    print(f"xb1 {xb1}") #[219 1026 1]

    xb2 = np.cross(np.cross(p2, p6), np.cross(p3, p7))
    xb2 = affinize(xb2)
    print(f"xb2 {xb2}") #[220 1012 1]

    xb3 = np.cross(np.cross(p1, p5), np.cross(p3, p7))
    xb3 = affinize(xb3)
    print(f"xb3 {xb3}") #[221 1018 1]

    xb = affinize(np.round((xb1 + xb2 + xb3)/3))
    print(f"xb  {xb}") #[220 1019 1]

    # racunamo tacku yb kao teziste tri tacke
    yb1 = np.cross(np.cross(p5, p6), np.cross(p7, p8))
    yb1 = affinize(yb1)
    print(f"yb1 {yb1}") #[765 -230 1]

    yb2 = np.cross(np.cross(p5, p6), np.cross(p2, p1))
    yb2 = affinize(yb2)
    print(f"yb2 {yb2}") #[801 -267 1]

    yb3 = np.cross(np.cross(p7, p8), np.cross(p2, p1))
    yb3 = affinize(yb3)
    print(f"yb3 {yb3}") #[779 -239 1]

    yb = affinize(np.round((yb1 + yb2 + yb3) / 3))
    print(f"yb  {yb}") #[782 -245 1]

    #racunamo nevidljivu tacku
    p4 = affinize(np.cross(np.cross(p8, xb), np.cross(p3, yb))) #[471 220 1]
   #p4 = affinize(np.cross(np.cross(p8, xb1), np.cross(p3, yb1))) #[470 221 1]
    print(f"p4  {p4}") #[316 156 1] tj. tacka (316, 156)
    return p4

#p1 = np.array([595, 301])
#p2 = np.array([292, 517])
#p3 = np.array([157, 379])
#p5 = np.array([665, 116])
#p6 = np.array([304, 295])
#p7 = np.array([135, 163])
#p8 = np.array([509, 43])

p1 = np.array([441, 202])
p2 = np.array([246, 456])
p3 = np.array([81, 359])
p5 = np.array([480, 57])
p6 = np.array([254, 285])
p7 = np.array([45, 190])
p8 = np.array([331, 23])

print(nevidljiva(p1, p2, p3, p5, p6, p7, p8)) #[316 156 1] tj. tacka (316, 15)
