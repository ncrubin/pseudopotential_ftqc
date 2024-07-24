import numpy as np
from pyscf.lib.numpy_helper import cartesian_prod

def lambdaloc(rloc, g1, g2, g3, n):
    # This is for computing the value of lambda for local pseudopotentials.
    # It gives the lambda in three parts, with the first needing to be multiplied by Z, 
    # the next by C_1, and the next by C_2.
    # The lattice vectors are given in g1, g2, g3, and n must be a vector [nx, ny, nz].

    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    lam1 = 0
    lam2 = 0
    lam3 = 0
    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    N1_vec = np.arange(-N1, N1 + 1)
    N2_vec = np.arange(-N2, N2 + 1)
    N3_vec = np.arange(-N3, N3 + 1)
    n_xyz = cartesian_prod([N1_vec, N2_vec, N3_vec])
    gmat = np.array([g1, g2, g3])
    vecs_t = n_xyz @ gmat
    nrms = np.einsum('ir,ir->i', vecs_t, vecs_t)
    rloc_exp = np.exp(-nrms * rloc**2 / 2, out=np.zeros_like(nrms), where=nrms!=0.)

    lam2 = np.sum(rloc_exp) 
    lam3 = np.sum(rloc_exp * np.abs(3 - nrms * rloc**2))
    lam1 = np.sum(np.divide(rloc_exp, nrms, out=np.zeros_like(nrms), where=nrms!=0.))

    # for nx in range(-N1, N1 + 1):
    #     for ny in range(-N2, N2 + 1):
    #         for nz in range(-N3, N3 + 1):
    #             mu = max([abs(nx), abs(ny), abs(nz)])
    #             if mu > 0:
    #                 vec = nx * g1 + ny * g2 + nz * g3
    #                 nrm = np.dot(vec, vec)
    #                 tmp = np.exp(-nrm * rloc**2 / 2)
    #                 lam1 = lam1 + tmp / nrm
    #                 lam2 = lam2 + tmp
    #                 lam3 = lam3 + tmp * abs(3 - nrm * rloc**2)

    tmp = np.sqrt(8 * np.pi**3) * rloc**3 / Omega
    lam1 = 4 * np.pi * lam1 / Omega
    lam2 = tmp * lam2
    lam3 = tmp * lam3

    return lam1, lam2, lam3
