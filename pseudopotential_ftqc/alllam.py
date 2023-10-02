import numpy as np
from pseudopotential_ftqc.lattice import lattice
from pseudopotential_ftqc.parameters import parameters
from pseudopotential_ftqc.lambdaloc import lambdaloc
from pseudopotential_ftqc.lamnonloc import lamnonloc

def alllam(n, types, nonu, lat, eta):
    """
    Provide the overall lambda value

    :param n: the vector for the size of the lattice [nx, ny, nz]
    :param types: the list of nucleus types, e.g. ["Al, "Ti"]
    :param nonu: a vector of the number of nuclei of each type
    :param lat: the lattice type (number should align with lattice in lattice.py) 
    :param eta: number of electrons
    """
    g1, g2, g3, d1, d2, d3 = lattice(lat)

    laml = 0
    lamnl = 0
    noty = len(types)
    
    for no in range(noty):
        # Get the pseudopotential parameters.
        Z, rl, C, r, E = parameters(types[no])
        
        # Compute parts of lambda for given rl.
        lam1, lam2, lam3 = lambdaloc(rl, g1, g2, g3, n)
        
        # Add together based on Z and C values. Multiply by the number of this type of nucleus, nonu(no).
        laml += (Z * lam1 + abs(C[0]) * lam2 + abs(C[1]) * lam3) * nonu[no]

        # Now for nonlocal lambda.
        lam0, lamm = lamnonloc(r, E, g1, g2, g3, d1, d2, d3, n)

        # The second output is for the case where we are accounting for maxima over boxes.
        lamnl += np.sum(lamm) * nonu[no]

    # Multiply by the number of electrons too.
    laml *= eta
    lamnl *= eta

    # Next compute lambda_T.
    v1 = (2**n[0] - 2) * g1
    v2 = (2**n[1] - 2) * g2
    v3 = (2**n[2] - 2) * g3
    lamT = (eta / 8) * max(np.linalg.norm(np.array([v1 + v2 + v3, v1 + v2 - v3, v1 - v2 + v3, v1 - v2 - v3]), axis=1))

    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    vsum = 0

    for nx in range(-N1, N1 + 1):
        for ny in range(-N2, N2 + 1):
            for nz in range(-N3, N3 + 1):
                tmp = np.linalg.norm(np.array([nx * g1 + ny * g2 + nz * g3]))
                if tmp > 0:
                    vsum += 1 / tmp

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))
    lamV = (2 * np.pi / Omega) * eta * (eta - 1) * vsum
    
    return laml, lamnl, lamT, lamV
