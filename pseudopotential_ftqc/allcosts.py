import numpy as np
import math
from pseudopotential_ftqc.parameters import parameters
from pseudopotential_ftqc.normcost import normcost

def allcosts(n, b, types, lat, order, pts, eta):
    nx = n[0]
    ny = n[1]
    nz = n[2]
    
    noty = len(types)

    lmax = 0
    ijmax = 0
    calM = 0

    for no in range(noty):
        # Get the pseudopotential parameters.
        Z, rl, C, r, E = parameters(types[no])

        # This finds the lmax value from the r vector.
        tmp = len(r) - 1
        if tmp > lmax:
            lmax = tmp
        
        # First the number of local parameters.
        calM = calM + 1 + sum(C != 0)
        
        # Now the number of nonlocal parameters.
        for m in range(len(r)):
            tmp = len(np.where(E[m] != 0)[0]) # sum(sum(E[m, :, :] != 0))
            tmp = int(math.sqrt(tmp))
            if tmp > ijmax:
                ijmax = tmp
            calM = calM + tmp * (tmp + 1) // 2

    # Start with base costs.
    cst = 7.75 * b**2

    # Now costs depending on the maximum of i,j coming from computing the polynomial in F.
    if ijmax == 1:
        cst = cst + 4 * b
    elif ijmax == 2:
        cst = cst + b**2 + 14 * b

    # Now add the cost of arithmetic for the Legendre polynomial with l=2.
    if lmax == 2:
        cst = cst + 1.5 * b**2

    # This adds more arithmetic costs for higher-order interpolation.
    # For example, order=2 is quadratic interpolation and gives an extra 2b^2 cost.
    # We also add the cost of the number of interpolation points here.
    cst = cst + order * b**2 + pts

    # This finds the cost of computing a norm for these Bravais vectors.
    ncst = normcost(n, b, lat)

    # This adds the cost of a single norm and a dot product for the pseudopotential arithmetic.
    # The cost of the dot product is twice the norm cost.
    cst = cst + 3 * ncst

    # Costs from list at the end of Section VI, page 21.

    # Cost 1
    # Now we compute the calligraphic M for the number of parameters in the pseudopotential.
    # calM is calculated above.
    cst = cst + 2 * max(n) * (3 * calM + 8)

    # Cost 2
    # Cost of preparing an equal superposition state for the number of electrons.
    neta = math.ceil(math.log2(eta))
    cst = cst + 14 * neta + 20

    # Cost 3
    # Cost of preparing an equal superposition for T.
    cst = cst + 6 * b

    # Cost 4
    # Complexity of swapping momentum registers.
    cst = cst + 4 * eta * sum(n) + 4 * eta - 8

    # Cost 5
    # Select cost for T.
    cst = cst + b

    # Cost 6
    # This is the cost of the amplitude amplification for preparing the 4/|k_nu| state.
    # We have three norms and a multiplication.
    cst = cst + 3 * ncst + 3 * b**2

    # Cost 7
    # The QROM cost for R was included in 1.

    # Cost 8
    cst = cst + 8 * sum(n)

    # Cost 9
    # The cost of the three multiplications needed for the phase shift according to the nuclear position.
    cst = cst + (nx**2 + ny**2 + nz**2) + 2 * b * (nx + ny + nz)

    return cst
