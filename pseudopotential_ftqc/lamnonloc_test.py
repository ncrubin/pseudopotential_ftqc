import numpy as np
from pseudopotential_ftqc.parameters import parameters
from pseudopotential_ftqc.lattice import lattice
from pseudopotential_ftqc.lamnonloc import (lamnonloc, 
                                            lamnonloc_from_maxt, 
                                            lambda_nonloc_nux_run, 
                                            lambda_nonloc_nux_nuy_nuz_run, 
                                            lamnonloc_from_nxnynz,
                                            )

def run_maxt_test(n, atom_type, lattice_type):
    g1, g2, g3, d1, d2, d3 = lattice(lattice_type)
    Z, rl, C, r_vec, E = parameters(atom_type)
    lam0, lamm = lamnonloc(r_vec, E, g1, g2, g3, d1, d2, d3, n, USE_MULTIPROCESSING=True)

    N1 = 2**n[0] - 1
    maxt_dict = {}
    for nut in range(1, 4 * N1 + 2):
        maxt_dict[nut] = lambda_nonloc_nux_run(nut=nut, n1=n[0], n2=n[1], n3=n[2], lattice_index=lattice_type, atom_type=atom_type,
                                                USE_MULTIPROCESSING=True, NUM_PROCESSORS=30, SAVE_MAXT=False)
    lam0_test, lamm_test = lamnonloc_from_maxt(r_vec, E, g1, g2, g3, d1, d2, d3, n, maxt_dict)
    assert np.allclose(lam0, lam0_test)
    assert np.allclose(lamm, lamm_test)


def test_non_loc_test():
    run_maxt_test([3, 3, 3], "Pd", 5)

def test_pd0_lam():
    run_maxt_test([1, 2, 3], "Pd", 5 )

def test_pd1_lam():
    run_maxt_test([2, 3, 3], "Pd", 5)

def test_limnnio_lam():
    run_maxt_test([2, 2, 3], "Li", 11)
    run_maxt_test([2, 2, 3], "Mn", 11)
    run_maxt_test([2, 2, 3], "Ni", 11)
    run_maxt_test([2, 2, 3], "O", 11)

def test_limnnio_2_lam():
    run_maxt_test([3, 4, 3], "Li", 11)
    run_maxt_test([3, 4, 3], "Mn", 11)
    run_maxt_test([3, 4, 3], "Ni", 11)
    run_maxt_test([3, 4, 3], "O", 11)

def test_pt_lamnl():
    run_maxt_test([2, 3, 3], "Pt", 7)

def test_pt_lamnl():
    run_maxt_test([2, 4, 3], "Rh", 9)

def test_xanadu_3_lam():
    run_maxt_test([2, 2, 3], "Li", 10)
    run_maxt_test([2, 2, 3], "Mn", 10)
    run_maxt_test([2, 2, 3], "F", 10)
    run_maxt_test([2, 2, 3], "O", 10)


def individual_run_nxnynz(n, atom_type, lattice_type):
    g1, g2, g3, d1, d2, d3 = lattice(lattice_type)
    Z, rl, C, r_vec, E = parameters(atom_type)
    lam0, lamm = lamnonloc(r_vec, E, g1, g2, g3, d1, d2, d3, n, USE_MULTIPROCESSING=False)
    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    maxt_dict = {}
    for nut in range(1, 4 * N1 + 2):
        for dy, nuy in enumerate(range(2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                maxt_dict[(nut, dy, dz)] =  lambda_nonloc_nux_nuy_nuz_run(nux_idx=nut, nuy_idx=dy, nuz_idx=dz,
                                                                          n1=n[0], n2=n[1], n3=n[2], 
                                                                          lattice_index=lattice_type,
                                                                          atom_type=atom_type,
                                                                          SAVE_RESULT=False
                                                                          )
    lam0_test, lamm_test = lamnonloc_from_nxnynz(r_vec=r_vec, 
                                                 E=E, 
                                                 g1=g1, 
                                                 g2=g2, 
                                                 g3=g3, 
                                                 d1=d1, 
                                                 d2=d2, 
                                                 d3=d3, 
                                                 n=n, 
                                                 maxt_dict=maxt_dict
                                                 )
    assert np.allclose(lam0, lam0_test)
    assert np.allclose(lamm, lamm_test)


def test_non_loc_test_nxyz():
    individual_run_nxnynz([3, 3, 3], "Pd", 5)

def test_pd0_lam_nxyz():
    individual_run_nxnynz([1, 2, 3], "Pd", 5 )

def test_pd1_lam_nxyz():
    individual_run_nxnynz([2, 3, 3], "Pd", 5)

def test_limnnio_lam_nyxz():
    individual_run_nxnynz([2, 2, 3], "Li", 11)
    individual_run_nxnynz([2, 2, 3], "Mn", 11)
    individual_run_nxnynz([2, 2, 3], "Ni", 11)
    individual_run_nxnynz([2, 2, 3], "O", 11)

def test_limnnio_2_lam_nyxz():
    individual_run_nxnynz([3, 4, 3], "Li", 11)
    individual_run_nxnynz([3, 4, 3], "Mn", 11)
    individual_run_nxnynz([3, 4, 3], "Ni", 11)
    individual_run_nxnynz([3, 4, 3], "O", 11)

def test_pt_lamnl_nyxz():
    individual_run_nxnynz([2, 3, 3], "Pt", 7)

def test_pt_lamnl_nxyz():
    individual_run_nxnynz([2, 4, 3], "Rh", 9)

def test_xanadu_3_lam_nxyz():
    individual_run_nxnynz([2, 2, 3], "Li", 10)
    individual_run_nxnynz([2, 2, 3], "Mn", 10)
    individual_run_nxnynz([2, 2, 3], "F", 10)
    individual_run_nxnynz([2, 2, 3], "O", 10)


if __name__ == "__main__":
    test_non_loc_test_nxyz()
    test_pd0_lam_nxyz()