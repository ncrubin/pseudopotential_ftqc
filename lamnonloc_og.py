import numpy as np


def lamnonloc(rl, E, g1, g2, g3, d1, d2, d3, n):
    # This is for computing the value of lambda for nonlocal pseudopotentials.
    # The lambda is the value in the tight case, but lamb is if we are using maxima based on the box for nu.

    N1 = 2**n[0] - 1
    N2 = 2**n[1] - 1
    N3 = 2**n[2] - 1
    # print(f"{N1=}", f"{N2=}", f"{N3=}")
    sz = E.shape
    assert len(sz) == 3
    # if len(sz) < 3:
    #     sz = (sz[0], sz[1], sz[1])

    # Determine scaled E.
    Esc = np.zeros((3, 3, 3))
    
    # Input GTH nonlocal pseudopotential projector parameters.
    Cli = np.array([
        [4 * np.sqrt(2), 8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105)],
        [8 / np.sqrt(3), 16 / np.sqrt(105), (32 / 3) / np.sqrt(1155)],
        [8 * np.sqrt(2 / 15), (16 / 3) * np.sqrt(2 / 105), (32 / 3) * np.sqrt(2 / 15015)]
    ]) * np.pi**(5/4)

    # Now multiply E by Cli.
    for el in range(sz[0]):
        for ii in range(sz[1]):
            for jj in range(sz[2]):
                Esc[el, ii, jj] = E[el, ii, jj] * Cli[el, ii] * Cli[el, jj] * rl[el]**(1 + 2 * (el + 1))

    Omega = (2 * np.pi)**3 / np.linalg.det(np.array([g1, g2, g3]))

    # First compute a full matrix of Fli values.
    Fli = np.zeros((sz[0], sz[1], 2 * N1 + 1, 2 * N2 + 1, 2 * N3 + 1))
    for nx in range(-N1, N1 + 1):
        for ny in range(-N2, N2 + 1):
            for nz in range(-N3, N3 + 1):
                vec = nx * g1 + ny * g2 + nz * g3
                nrm = np.dot(vec, vec)
                
                # l=0
                tmp1 = nrm * rl[0] ** 2
                tmp2 = np.exp(-tmp1 / 2)
                tmp = np.array([1, 3 - tmp1, 15 - 10 * tmp1 + tmp1 ** 2]) * tmp2
                for jj in range(sz[1]):
                    Fli[0, jj, nx + N1, ny + N2, nz + N3] = tmp[jj]
                
                # l=1
                if sz[0] > 1:
                    tmp1 = nrm * rl[1] ** 2
                    tmp2 = np.exp(-tmp1 / 2)
                    tmp = np.array([1, 5 - tmp1, 35 - 14 * tmp1 + tmp1 ** 2]) * tmp2
                    for jj in range(sz[1]):
                        Fli[1, jj, nx + N1, ny + N2, nz + N3] = tmp[jj]
                
                # l=2
                if sz[0] > 2:
                    tmp1 = nrm * rl[2] ** 2
                    tmp2 = np.exp(-tmp1 / 2)
                    tmp = np.array([1, 7 - tmp1, 63 - 18 * tmp1 + tmp1 ** 2]) * tmp2
                    for jj in range(sz[1]):
                        Fli[2, jj, nx + N1, ny + N2, nz + N3] = tmp[jj] 

    # #########################
    # # remove after done
    # import scipy.io
    # Fli_test = scipy.io.loadmat('Fli_val.mat')['Fli']
    # assert np.allclose(Fli_test, Fli)
    # #############################

    maxs = np.zeros((4 * N1 + 1, sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
    # We loop over all nu values, but only use non-negative nu_y because it must be symmetric under reflection about all three.
    for nut in range(1, 4 * N1 + 2):
        nux = nut - 2 * N1 - 1
        maxt = np.zeros((sz[0], sz[1], sz[2], 2 * N2 + 1, 4 * N3 + 1))
        
        for nuy in range(2 * N2 + 1):
            for nuz in range(-2 * N3, 2 * N3 + 1):
                # Now find appropriate range of p vector given nu.
                for mx in range(max(-N1, -nux - N1), min(N1, -nux + N1) + 1):
                    for my in range(max(-N2, -nuy - N2), min(N2, -nuy + N2) + 1):
                        for mz in range(max(-N3, -nuz - N3), min(N3, -nuz + N3) + 1):
                            nx = nux + mx
                            ny = nuy + my
                            nz = nuz + mz
                            vecp = nx * g1 + ny * g2 + nz * g3
                            vecq = mx * g1 + my * g2 + mz * g3
                            dot = np.dot(vecp, vecq)
                            # print(f"{nux=}", f"{nuy=}", f"{nuz=}", 
                            #       f"{mx=}", f"{my=}", f"{mz=}", 
                            #       f"{dot=}")
                            Pl = np.array([1, dot, (3 * dot**2 - np.dot(vecp, vecp) * np.dot(vecq, vecq)) / 2])
                            
                            for el in range(1, sz[0] + 1):
                                for ii in range(1, sz[1] + 1):
                                    for jj in range(1, sz[2] + 1):
                                        if Esc[el - 1, ii - 1, jj - 1] != 0:
                                            tmp = abs((2 * el - 1) / (4 * np.pi * Omega) * Pl[el - 1] * Esc[el - 1, ii - 1, jj - 1] * Fli[el - 1, ii - 1, nx + N1, ny + N2, nz + N3] * Fli[el - 1, jj - 1, mx + N1, my + N2, mz + N3])
                                            if tmp > maxt[el - 1, ii - 1, jj - 1, nuy, nuz + 2 * N3]:
                                                maxt[el - 1, ii - 1, jj - 1, nuy, nuz + 2 * N3] = tmp
        
        maxs[nut - 1, :, :, :, :, :] = maxt[:, :, :, :, :]

    # #########################
    # # Remove after test
    # maxs_test = scipy.io.loadmat('maxs_val.mat')['maxs']
    # assert np.allclose(maxs_test, maxs)
    # ############################

    lambda_val = np.zeros((sz[0], sz[1], sz[2]))
    mumax = max(n[0] + d1, n[1] + d2, n[2] + d3) + 2
    maxy_val = np.zeros((sz[0], sz[1], sz[2], mumax))
    nums = np.zeros(mumax)
    D1 = 2**d1
    D2 = 2**d2
    D3 = 2**d3

    for nux in range(-2 * N1, 2 * N1 + 1):
        for nuy in range(-2 * N2, 2 * N2 + 1):
            for nuz in range(-2 * N3, 2 * N3 + 1):
                mut = max(abs(nux * D1), abs(nuy * D2), abs(nuz * D3))
                if mut == 0:
                    mu = 1
                else:
                    mu = int(np.floor(np.log2(mut))) + 2
                
                nums[mu - 1] += 1
                
                for el in range(sz[0]):
                    for ii in range(sz[1]):
                        for jj in range(sz[2]):
                            if Esc[el, ii, jj] != 0:
                                # print(maxs.shape)
                                # tmp = maxs[nux + 2 * N1, el, ii, jj, abs(nuy) + 1, nuz + 2 * N3 + 1]
                                tmp = maxs[nux + 2 * N1, el, ii, jj, abs(nuy), nuz + 2 * N3]
                                lambda_val[el, ii, jj] += tmp
                                if tmp > maxy_val[el, ii, jj, mu - 1]:
                                    maxy_val[el, ii, jj, mu - 1] = tmp
    
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