import os

try:
    mkl_num_threads = os.environ.get('MKL_NUM_THREADS')
    os.environ['MKL_NUM_THREADS'] = mkl_num_threads
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = omp_num_threads
except (KeyError, NameError, TypeError):
    os.environ['MKL_NUM_THREADS'] = str(1)
    os.environ['OMP_NUM_THREADS'] = str(1)

import multiprocessing
import numpy as np
from pyscf.lib import direct_sum, cartesian_prod
import time

from pseudopotential_ftqc.parameters import parameters
from pseudopotential_ftqc.lattice import lattice

def compute_Esc(sz, Cli, E, rl):
    # Determine scaled E.
    Esc = np.zeros((3, 3, 3))
    # Now multiply E by Cli.
    for el in range(sz[0]):
        for ii in range(sz[1]):
            for jj in range(sz[2]):
                Esc[el, ii, jj] = E[el, ii, jj] * Cli[el, ii] * Cli[el, jj] * rl[el]**(1 + 2 * (el + 1))
    return Esc


def compute_Fli(sz, N1, N2, N3, g1, g2, g3, rl):
    # First compute a full matrix of Fli values.
    Fli = np.zeros((sz[0], sz[1], 2 * N1 + 1, 2 * N2 + 1, 2 * N3 + 1))
    N1_vec = np.arange(-N1, N1 + 1)
    N2_vec = np.arange(-N2, N2 + 1)
    N3_vec = np.arange(-N3, N3 + 1)
    kx_vec = np.outer(N1_vec, g1)
    ky_vec = np.outer(N2_vec, g2)
    kz_vec = np.outer(N3_vec, g3)
    kvecs = direct_sum('xr+yr+zr->xyzr', kx_vec, ky_vec, kz_vec)
    nrms = np.einsum('...r,...r', kvecs, kvecs)
    l0_tmp1 = nrms * (rl[0]**2)
    l0_tmp2 = np.exp(-l0_tmp1 / 2)
    if sz[0] > 1:
        l1_tmp1 = nrms * (rl[1]**2)
        l1_tmp2 = np.exp(-l1_tmp1 / 2)
    if sz[0] > 2:
        l2_tmp1 = nrms * (rl[2]**2)
        l2_tmp2 = np.exp(-l2_tmp1 / 2)
    for jj in range(sz[1]):
        if jj == 0:
            Fli[0, 0, :, :, :] = l0_tmp2
            if sz[0] > 1:
                Fli[1, 0, :, :, :] = l1_tmp2
            if sz[0] > 2:
                Fli[2, 0, :, :, :] = l2_tmp2
        elif jj == 1:
            Fli[0, 1, :, :, :] = (3 - l0_tmp1) * l0_tmp2
            if sz[0] > 1:
                Fli[1, 1, :, :, :] = (5 - l1_tmp1) * l1_tmp2
            if sz[0] > 2:
                Fli[2, 1, :, :, :] = (7 - l2_tmp1) * l2_tmp2
        elif jj == 2:
            Fli[0, 2, :, :, :] = (15 - 10 * l0_tmp1 + l0_tmp1**2) * l0_tmp2
            if sz[0] > 1:
                Fli[1, 2, :, :, :] = (35 - 14 * l1_tmp1 + l1_tmp1**2) * l1_tmp2
            if sz[0] > 2:
                Fli[2, 2, :, :, :] = (63 - 18 * l2_tmp1 + l2_tmp1**2) * l2_tmp2
    return Fli

def post_process_maxs(sz, n, N1, N2, N3, Esc, maxs, d1, d2, d3):
    lambda_val = np.zeros((sz[0], sz[1], sz[2]))
    lambda_val_ncr = np.zeros((sz[0], sz[1], sz[2]))
    mumax = max(n[0] + d1, n[1] + d2, n[2] + d3) + 2
    maxy_val = np.zeros((sz[0], sz[1], sz[2], mumax))
    nums = np.zeros(mumax)
    D1 = 2**d1
    D2 = 2**d2
    D3 = 2**d3

    nux_vals = np.arange(-2 * N1, 2 * N1 + 1, dtype=int)
    nuy_vals = np.arange(-2 * N2, 2 * N2 + 1, dtype=int)
    nuz_vals = np.arange(-2 * N3, 2 * N3 + 1, dtype=int)

    mut_xyz = np.zeros((4 * N1 + 1, 4 * N2 + 1, 4 * N3 + 1), dtype=int)
    d1_nux_vals = np.abs(D1 * nux_vals)
    d2_nuy_vals = np.abs(D2 * nuy_vals)
    d3_nuz_vals = np.abs(D3 * nuz_vals)
    for dx, dnux in enumerate(d1_nux_vals):
        for dy, dnuy in enumerate(d2_nuy_vals):
            for dz, dnuz in enumerate(d3_nuz_vals):
                mut_xyz[dx, dy, dz] = max(dnux, dnuy, dnuz)

    zero_mask = mut_xyz != 0
    mu_xyz = np.floor(np.log2(mut_xyz, where=zero_mask, out=np.ones_like(mut_xyz, dtype=float)))
    mu_xyz = np.asarray(mu_xyz, dtype=int) 
    mu_xyz = np.where(zero_mask, mu_xyz + 2, np.ones_like(mut_xyz, dtype=int))

    dx_nux_vals = nux_vals + 2 * N1
    dy_nuy_vals = np.abs(nuy_vals)
    dz_nuz_vals = nuz_vals + 2 * N3


    Esc_mask = Esc[:sz[0], :sz[1], :sz[2]] != 0.
    tmaxs = maxs[:, :sz[0], :sz[1], :sz[2], :, :]

    nums_ncr = np.zeros_like(nums)
    for dx, nux in enumerate(range(-2 * N1, 2 * N1 + 1)):
        for dy, nuy in enumerate(range(-2 * N2, 2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                nums_ncr[mu_xyz[dx, dy, dz] - 1] += 1

    abs_dy = slice(0, 2 * N2 + 1)
    outer_ncr_tmp1 = np.einsum('xlijyz,lij->xlijyz', tmaxs[:, :, :, :, abs_dy, :], Esc_mask)
    lambda_val_ncr = np.sum(outer_ncr_tmp1, axis=(0,4,5))

    abs_dy = slice(1, 2 * N2 + 1) # don't count zero twice!
    outer_ncr_tmp2 = np.einsum('xlijyz,lij->xlijyz', tmaxs[:, :, :, :, abs_dy, :], Esc_mask)
    lambda_val_ncr += np.sum(outer_ncr_tmp2, axis=(0,4,5))

    for dx, nux in enumerate(range(-2 * N1, 2 * N1 + 1)):
        for dy, nuy in enumerate(range(-2 * N2, 2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                mu_idx = mu_xyz[dx, dy, dz] - 1
                ncr_tmp = outer_ncr_tmp1[dx, :, :, :, dy_nuy_vals[dy], dz]
                mask = ncr_tmp > maxy_val[:, :, :, mu_idx]
                test_maxy_val = np.where(mask, ncr_tmp, maxy_val[:, :, :, mu_idx])
                maxy_val[:, :, :, mu_idx] = test_maxy_val

    # assert np.allclose(lambda_val_ncr, lambda_val)
    lambda_val = lambda_val_ncr

    nums = nums_ncr

    for el in range(sz[0]):
        for ii in range(sz[1]):
            for jj in range(ii + 1, sz[2]):
                lambda_val[el, ii, jj] = lambda_val[el, jj, ii]
                maxy_val[el, ii, jj, :] = maxy_val[el, jj, ii, :]
    
    lamb = np.zeros((sz[0], sz[2], sz[1]))
    
    for el in range(sz[0]):
        for ii in range(sz[1]):
            for jj in range(sz[2]):
                lamb[el, jj, ii] = np.sum(nums * maxy_val[el, ii, jj, :])
    
    return lambda_val, lamb


def inner_loop_q(sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift, dy=None, dz=None):
    # THIS IS ALL FOR VECTORIZING mx, my, mz
    # Calculate ranges for p vectors
    min_mx = max(-N1, -nux - N1)
    max_mx = min(N1, -nux + N1)
    min_my = max(-N2, -nuy - N2)
    max_my = min(N2, -nuy + N2)
    min_mz = max(-N3, -nuz - N3)
    max_mz = min(N3, -nuz + N3)

    # Calculate dot products and Pl coefficients
    mx_vec = np.arange(min_mx, max_mx + 1)
    my_vec = np.arange(min_my, max_my + 1)
    mz_vec = np.arange(min_mz, max_mz + 1)
    kqx_vec = np.outer(mx_vec, g1)
    kqy_vec = np.outer(my_vec, g2)
    kqz_vec = np.outer(mz_vec, g3)
    kpx_vec = np.outer(mx_vec + nux, g1)
    kpy_vec = np.outer(my_vec + nuy, g2)
    kpz_vec = np.outer(mz_vec + nuz, g3)
    kpvecs = direct_sum('xr+yr+zr->xyzr', kpx_vec, kpy_vec, kpz_vec)
    kqvecs = direct_sum('xr+yr+zr->xyzr', kqx_vec, kqy_vec, kqz_vec)
    cross_dots = np.einsum('...r,...r', kpvecs, kqvecs)
    dots_p = np.einsum('...r,...r', kpvecs, kpvecs)
    dots_q = np.einsum('...r,...r', kqvecs, kqvecs)
    Plvec = np.ones((3,) + dots_p.shape) 
    Plvec[1] = cross_dots
    Plvec[2] = (3 * cross_dots**2 - dots_p * dots_q) / 2.

    # slices for taking appropriate slices of Fli
    mx_slice = slice(min_mx + N1, max_mx + N1 + 1, 1)
    my_slice = slice(min_my + N2, max_my + N2 + 1, 1)
    mz_slice = slice(min_mz + N3, max_mz + N3 + 1, 1)
    nx_slice = slice(min_mx + nux + N1, max_mx + nux + N1 + 1, 1)
    ny_slice = slice(min_my + nuy + N2, max_my + nuy + N2 + 1, 1)
    nz_slice = slice(min_mz + nuz + N3, max_mz + nuz + N3 + 1, 1)

    # big multiplication
    ncr_tmp_vec = np.abs(np.einsum('l,lxyz,lij,lixyz,ljxyz->lijxyz',
                        legendre_shift,
                        Plvec[:sz[0], :, :, :],
                        Esc[:sz[0], :sz[1], :sz[2]], 
                        Fli[:sz[0], :sz[1], nx_slice, ny_slice, nz_slice],
                        Fli[:sz[0], :sz[2], mx_slice, my_slice, mz_slice], optimize=['einsum_path', (0, 1, 2, 3, 4)],
                        )
                        )
    # max over mx, my, mz 
    max_val = np.max(ncr_tmp_vec, axis=(3, 4, 5))
    if dy == None or dz == None:
        return max_val
    return max_val, dy, dz

def lamnonloc(r_vec, E, g1, g2, g3, d1, d2, d3, n, USE_MULTIPROCESSING=False, NUM_PROCESSORS=30):
    # This is for computing the value of lambda for nonlocal pseudopotentials.
    # The lambda is the value in the tight case, but lamb is if we are using maxima based on the box for nu.
    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    sz = E.shape
    assert len(sz) == 3
   
    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)

    Esc = compute_Esc(sz, Cli, E, r_vec)

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    Fli = compute_Fli(sz, N1, N2, N3, g1, g2, g3, r_vec)

    maxs = np.zeros((4 * N1 + 1, sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    maxt = np.zeros((sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    legendre_shift = (2 * np.arange(1, sz[0] + 1) - 1) / (4 * np.pi * Omega)

    # We loop over all nu values, but only use non-negative nu_y because it must be symmetric under reflection about all three.
    for nut in range(1, 4 * N1 + 2):
        nux = nut - 2 * N1 - 1
        maxt.fill(0.)
        if USE_MULTIPROCESSING:
            # Generate input for starmap
            starmap_inputs = []
            for dy, nuy in enumerate(range(2 * N2 + 1)):
                for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                    starmap_inputs.append((sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift, dy, dz))
            with multiprocessing.Pool(processes=NUM_PROCESSORS) as pool:
                results = pool.starmap(inner_loop_q, starmap_inputs)
            for res, dy, dz in results:
                maxt[:, :, :, dy, dz] = res
            maxs[nut - 1, :, :, :, :, :] = maxt[:, :, :, :, :]
        else:
            for dy, nuy in enumerate(range(2 * N2 + 1)):
                for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                    maxt[:, :, :, dy, dz] = inner_loop_q(sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift)
            maxs[nut - 1, :, :, :, :, :] = maxt[:, :, :, :, :]

    lambda_val, lamb = post_process_maxs(sz, n, N1, N2, N3, Esc, maxs, d1, d2, d3)
   
    return lambda_val, lamb

def lambda_nonloc_nux_run(nut: int, n1: int, n2: int, n3: int, lattice_index: int, atom_type: str,
                          USE_MULTIPROCESSING=True, NUM_PROCESSORS=30, SAVE_MAXT=False, debug=False):
    n = [n1, n2, n3]
    g1, g2, g3, d1, d2, d3 = lattice(lattice_index)
    Z, rl, C, r_vec, E = parameters(atom_type)

    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    sz = E.shape
    assert len(sz) == 3

    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)

    Esc = compute_Esc(sz, Cli, E, r_vec)

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    start_time = time.time()
    Fli = compute_Fli(sz, N1, N2, N3, g1, g2, g3, r_vec)
    end_time = time.time()
    if debug:
        print("compute_Fli runtime = ", f"{(end_time - start_time)=}")

    maxt = np.zeros((sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    legendre_shift = (2 * np.arange(1, sz[0] + 1) - 1) / (4 * np.pi * Omega)

    nux = nut - 2 * N1 - 1
    if USE_MULTIPROCESSING:
        # Generate input for starmap
        starmap_inputs = []
        start_time = time.time()
        for dy, nuy in enumerate(range(2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                starmap_inputs.append((sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift, dy, dz))
        end_time = time.time()
        if debug:
            print("starmap input gen runtime ", f"{(end_time - start_time)=}")
        
        start_time = time.time()
        with multiprocessing.Pool(processes=NUM_PROCESSORS) as pool:
            results = pool.starmap(inner_loop_q, starmap_inputs)
        end_time = time.time()
        if debug:
            print("multiprocess runtime ", f"{(end_time - start_time)=}")

        start_time = time.time()
        for res, dy, dz in results:
            maxt[:, :, :, dy, dz] = res
        end_time = time.time()
        if debug:
            print("result storage runtime ", f"{(end_time - start_time)=}")

    else:
        for dy, nuy in enumerate(range(2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                maxt[:, :, :, dy, dz] = inner_loop_q(sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift)

    if SAVE_MAXT:
        np.save("maxt_nut_{}_n_{}{}{}_atom_type_{}_lattice_{}.npy".format(nut, n1, n2, n3, atom_type, lattice_index), maxt)

    return maxt

def lamnonloc_from_maxt(r_vec, E, g1, g2, g3, d1, d2, d3, n, maxt_dict):
    # This is for computing the value of lambda for nonlocal pseudopotentials.
    # The lambda is the value in the tight case, but lamb is if we are using maxima based on the box for nu.
    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    sz = E.shape
    assert len(sz) == 3
   
    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)

    Esc = compute_Esc(sz, Cli, E, r_vec)

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    Fli = compute_Fli(sz, N1, N2, N3, g1, g2, g3, r_vec)

    maxs = np.zeros((4 * N1 + 1, sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    legendre_shift = (2 * np.arange(1, sz[0] + 1) - 1) / (4 * np.pi * Omega)

    inner_loop_times = []

    for nut in range(1, 4 * N1 + 2):
        nux = nut - 2 * N1 - 1
        maxs[nut - 1, :, :, :, :, :] = maxt_dict[nut]

    lambda_val, lamb = post_process_maxs(sz, n, N1, N2, N3, Esc, maxs, d1, d2, d3)
   
    return lambda_val, lamb



def lambda_nonloc_nux_nuy_nuz_run(nux_idx: int, nuy_idx: int, nuz_idx: int,
                                  n1: int, n2: int, n3: int,
                                  lattice_index: int,
                                  atom_type: str,
                                  SAVE_RESULT=False
                                  ):
    n = [n1, n2, n3]
    g1, g2, g3, d1, d2, d3 = lattice(lattice_index)
    Z, rl, C, r_vec, E = parameters(atom_type)

    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    sz = E.shape
    assert len(sz) == 3

    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)

    Esc = compute_Esc(sz, Cli, E, r_vec)

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    Fli = compute_Fli(sz, N1, N2, N3, g1, g2, g3, r_vec)
    nux = nux_idx - 2 * N1 - 1
    nuy = nuy_idx
    nuz = nuz_idx - 2 * N3
    legendre_shift = (2 * np.arange(1, sz[0] + 1) - 1) / (4 * np.pi * Omega)
    result = inner_loop_q(sz, Esc, Fli, nux, nuy, nuz, N1, N2, N3, g1, g2, g3, legendre_shift)
    if SAVE_RESULT:
        np.save("nx_{}_ny_{}_nz_{}_n_{}{}{}_atom_type_{}_lattice_{}.npy".format(nux_idx, nuy_idx, nuz_idx, n1, n2, n3, atom_type, lattice_index), result)
    return result


def lamnonloc_from_nxnynz(r_vec, E, g1, g2, g3, d1, d2, d3, n, maxt_dict):
    # This is for computing the value of lambda for nonlocal pseudopotentials.
    # The lambda is the value in the tight case, but lamb is if we are using maxima based on the box for nu.
    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    sz = E.shape
    assert len(sz) == 3
    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)
    Esc = compute_Esc(sz, Cli, E, r_vec)
    maxs = np.zeros((4 * N1 + 1, sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    for nut in range(1, 4 * N1 + 2):
        nux = nut - 2 * N1 - 1
        for dy, nuy in enumerate(range(2 * N2 + 1)):
            for dz, nuz in enumerate(range(-2 * N3, 2 * N3 + 1)):
                maxs[nut - 1, :, :, :, dy, dz] = maxt_dict[(nut, dy, dz)]

    lambda_val, lamb = post_process_maxs(sz, n, N1, N2, N3, Esc, maxs, d1, d2, d3)
   
    return lambda_val, lamb




if __name__ == "__main__":
    from pseudopotential_ftqc.parameters import parameters
    from pseudopotential_ftqc.lattice import lattice
    import pickle
    # single core performance
    # 5, 5, 5 is 0.1 seconds per inner q-sum , 7875 for N2, N3 vals, over 30 cores this is 0.6 hours
    # 6, 6, 6 is 1 seconds per inner q-sum, 32131 for N2, N3 vals over 30 cores this is 2.5 hours
    # 7, 7, 7 is 8 seconds per inner q-sum, 129795 for N2, N3 vals, over 30 cores this is 9 hours
    # 6, 7, 8 is 8 seconds per inner q-sum, 260355 for N2, N3 vals, over 30 cores this is 20 hours
    n = [7, 6, 6]
    # N = 2**np.array(n) - 1
    # total_num_inner_loops = (2 * N[1] + 1) * (4 * N[2] + 1)
    # g1, g2, g3, d1, d2, d3 = lattice(5)
    # Z, rl, C, r_vec, E = parameters("Pd")
    # USE_MULTIPROCESSING = True
    # NUM_PROCESSORS = 30
    start_time = time.time()
    # lam0, lamm = lamnonloc(r_vec, E, g1, g2, g3, d1, d2, d3, n, USE_MULTIPROCESSING=True, NUM_PROCESSORS=30)
    # end_time = time.time()
    # print(f"{(end_time - start_time)=}")

    lattice_type = 5
    atom_type = "Pd"
    g1, g2, g3, d1, d2, d3 = lattice(lattice_type)
    Z, rl, C, r_vec, E = parameters(atom_type)
    N1 = 2**n[0] - 1
    # for nut in range(1, 4 * N1 + 2):
    nut = (4 * N1 + 2) // 2
    start_time = time.time()
    res = lambda_nonloc_nux_run(nut=nut, n1=n[0], n2=n[1], n3=n[2], lattice_index=lattice_type, atom_type=atom_type,
                                                USE_MULTIPROCESSING=True, NUM_PROCESSORS=60, SAVE_MAXT=False, debug=True)
    end_time = time.time()
    print(f"{(end_time - start_time)=}")
    exit()


    # g1, g2, g3, d1, d2, d3 = lattice(lattice_type)
    # Z, rl, C, r_vec, E = parameters(atom_type)
    # start_time1 = time.time()
    # maxt_dict = {}
    # for nut in range(1, 4 * N1 + 2):
    #     maxt_dict[nut] = lambda_nonloc_nux_run(nut=nut, n1=n[0], n2=n[1], n3=n[2], lattice_index=lattice_type, atom_type=atom_type,
    #                                             USE_MULTIPROCESSING=True, NUM_PROCESSORS=30, SAVE_MAXT=False)
    # end_time1 = time.time()
    # print(f"{(end_time1 - start_time1)=}")
    # with open('maxt_dict_{}{}{}.pickle'.format(n[0], n[1], n[2]), 'wb') as handle:
    #     pickle.dump(maxt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # exit()
    # with open('maxt_dict_{}{}{}.pickle'.format(n[0], n[1], n[2]), 'rb') as handle:
    #     maxt_dict = pickle.load(handle)

    # start_time2 = time.time()
    # lam0_test, lamm_test = lamnonloc_from_maxt(r_vec, E, g1, g2, g3, d1, d2, d3, n, maxt_dict)
    # end_time2 = time.time()
    # print(f"{(end_time2 - start_time2)=}")
