from allcosts import allcosts
from alllam import alllam
import numpy as np
import time

if __name__ == "__main__":
    [laml, lamnl, lamT, lamV] = alllam([2, 1, 3],["Li", "Mn", "F", "O"],[12, 16, 16, 32], 10, 428)
    assert np.isclose(laml, 84191.9137306989287)
    assert np.isclose(lamT, 237.5836178849430)
    assert np.isclose(lamV, 36535.0271848313787)
    assert np.isclose(lamnl, 261975.9651466767536)

    [laml, lamnl, lamT, lamV] = alllam([2, 4, 3],["Li", "Mn", "O"],[8, 16, 48], 10, 408)
    assert np.isclose(laml, 262123.7272977028333)
    assert np.isclose(lamT, 264.7284113020364)
    assert np.isclose(lamV, 231827.6938349053089)
    assert np.isclose(lamnl, 2797454.0732959737070)

    # cst = allcosts([1, 2, 3],20,["Li", "Mn", "Ni", "O"],11,1,256, 468)
    [laml, lamnl, lamT, lamV] = alllam([1, 2, 3],["Pd"],[27],5,270)
    assert np.isclose(laml, 47531.7856667841552)
    assert np.isclose(lamT, 55.5772116413959)
    assert np.isclose(lamV, 21397.2076705073378 )
    assert np.isclose(lamnl, 282327.3900572554558)


    # [laml, lamnl, lamT, lamV] = alllam([2, 3, 3],["Pd"],[27],5,270)
    # assert np.isclose(laml, 82650.7642178183887)
    # assert np.isclose(lamT, 120.5251884322382)
    # assert np.isclose(lamV, 65249.9765359183803)
    # assert np.isclose(lamnl, 1366687.1751366390381)

    # start_time = time.time()
    #                                             # Li 22, Mn 14, Ni 6, O 47
    # [laml, lamnl, lamT, lamV] = alllam([2, 2, 3],["Li", "Mn", "Ni", "O"],[22, 14, 6, 47], 11, 468)
    # print(laml)
    # print(lamT)
    # print(lamV)
    # print(lamnl)
    # # assert np.isclose(laml, 243781.6282627519395)
    # # assert np.isclose(lamT, 128.8705226502210)
    # # assert np.isclose(lamV, 127864.2257922822901)
    # # assert np.isclose(lamnl, 1498956.0633884919807)
    # end_time = time.time()
    # print(f"{(end_time - start_time)=}")

    [laml, lamnl, lamT, lamV] = alllam([1, 2, 3],["Li", "Mn", "Ni", "O"],[22, 14, 6, 47], 11, 468)
    assert np.isclose(laml, 185483.6047614611452)
    assert np.isclose(lamT, 79.2860339737204)
    assert np.isclose(lamV, 73472.1677942285169)
    assert np.isclose(lamnl, 656508.7416548251640)

    # start_time = time.time()
    # [laml, lamnl, lamT, lamV] = alllam([2, 3, 3],["Pt"],[27], 7,270)
    # assert np.isclose(laml, 102356.5721002650535)
    # assert np.isclose(lamT, 122.3578028482575)
    # assert np.isclose(lamV, 69957.7833838449296)
    # assert np.isclose(lamnl, 1857483.6096443952993)
    # end_time = time.time()
    # print(f"{(end_time - start_time)=}")

    # [laml, lamnl, lamT, lamV] = alllam([2, 4, 1],["Pt"],[27], 7,270)
    # assert np.isclose(laml, 39861.2889253929025)
    # assert np.isclose(lamT, 232.7935267744722)
    # assert np.isclose(lamV, 23567.8664826328750)
    # assert np.isclose(lamnl, 428734.0797997858608)


    # [laml, lamnl, lamT, lamV] = alllam([2, 3, 3],["Rh"],[27], 9,243)
    # assert np.isclose(laml, 69523.9800708498369)
    # assert np.isclose(lamT, 111.3436306607220)
    # assert np.isclose(lamV, 55186.5632881460842)
    # assert np.isclose(lamnl, 1749850.3495306812692)

    # [laml, lamnl, lamT, lamV] = alllam([2, 4, 1],["Rh"],[27], 9,243)
    # assert np.isclose(laml, 27751.5660195487108)
    # assert np.isclose(lamT, 215.9513469847757)
    # assert np.isclose(lamV, 18080.9129221698167)
    # assert np.isclose(lamnl, 445778.5642212145030)

    


