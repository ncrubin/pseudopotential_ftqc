import numpy as np
    
def normcost(n = None,b = None,num = None): 
    # Gives the cost of computing the norm of a vector using Bravais vectors.
    # The input n is a three-component vector [nx,ny,nz], b is the number of
    # bits of precision, and num gives the number of the lattice according to
    # the listing:
    # 1 - LiNo2 (C2/m)
    # 2 - LiNo2 (P21/c)
    # 3 - LiNo2 (P2/c)
    # 4 - LiNo2 (R3m)
    # 5 - Pd (3x3)
    # 6 - Pt (2x2)
    # 7 - Pt (3x3)
    # 8 - Pt (4x4)
    # 9 - Rh (3x3)
    # 10 - LiMnO3
    # 11 - Li[LiNiMn]O2
    # 12 - LiMnO2F
    # 13 - C
    # 14 - AlN
    # 15 - CaTiO3
    
    nx = n[0]
    ny = n[1]
    nz = n[2]
    if num == 1:
        nxy = np.amax([nx,ny])
        cst = nxy ** 2 + nz ** 2 + 2 * nx * ny + 2 * nxy * nz
        cst = cst + 2 * nxy * (nxy + b) + 2 * nz * (nz + b) + (nx + ny) ** 2 / 2 + (nx + ny) * b
    elif num == 4:
        nxyz = np.amax(n)
        cst = nxyz ** 2 + 2 * nx * ny + 2 * np.amax(ny,nx) * nz
        cst = cst + 2 * nxyz * (nxyz + b)
    elif (num == 2) or (num == 3) or (num == 10):
        cst = (nx + nz) ** 2 + ny ** 2
        cst = cst + 2 * (nx ** 2 + ny ** 2 + nz ** 2) + 2 * b * (nx + ny + nz)
    elif (num == 11):
        cst = (nx + ny) ** 2 + nz ** 2
        cst = cst + 2 * (nx ** 2 + ny ** 2 + nz ** 2) + 2 * b * (nx + ny + nz)
    elif (num == 5) or (num == 6) or (num == 7) or (num == 8) or (num == 9) or (num == 14):
        cst = np.amax([nx,ny]) ** 2 + nz ** 2 + 2 * nx * ny + 2 * nz * (nz + b)
    elif (num == 12):
        cst = (nx ** 2 + ny ** 2 + nz ** 2) + 2 * nx * (nx + b)
    elif (num == 15):
        cst = (nx ** 2 + ny ** 2 + nz ** 2) + 2 * nx * (nx + b) + 2 * ny * (ny + b)
    elif (num == 13):
        cst = 3 * np.amax(n) ** 2
    
    return cst