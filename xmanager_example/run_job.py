import os
import sys

from absl import app, flags, logging
from pseudopotential_ftqc.lamnonloc import lambda_nonloc_nux_run
from pseudopotential_ftqc.lattice import lattice
from pseudopotential_ftqc.parameters import parameters

_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "output directory")
_NUT = flags.DEFINE_integer("nut", 0, help="x")
_N1 = flags.DEFINE_integer("n1", 0, help="y")
_N2 = flags.DEFINE_integer("n2", 0, help="z")
_N3 = flags.DEFINE_integer("n3", 0, help="x")
_LATTICE_TYPE = flags.DEFINE_integer("lattice", 0, help="x")
_ATOM_TYPE = flags.DEFINE_string("atom", None, help="x")

def run_nux_run(nut, atom_type, lattice_type, n):
    mkl_num_threads = os.environ.get('MKL_NUM_THREADS')
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    logging.info(f"MKL = {mkl_num_threads=}")
    logging.info(f"OMP = {omp_num_threads=}")
    g1, g2, g3, d1, d2, d3 = lattice(lattice_type)
    Z, rl, C, r_vec, E = parameters(atom_type)
    lambda_nonloc_nux_run(nut=nut,
    n1=n[0],
    n2=n[1],
    n3=n[2],
    lattice_index=lattice_type,
    atom_type=atom_type,
    USE_MULTIPROCESSING=True,
    NUM_PROCESSORS=int(mkl_num_threads),
    SAVE_MAXT=True,
    PATH=_OUTPUT_DIR.value,
    )

def main(_):
    nut = _NUT.value
    n1 = _N1.value
    n2 = _N2.value
    n3 = _N3.value
    lattice_type = _LATTICE_TYPE.value 
    atom_type = _ATOM_TYPE.value 
    # I guess this is probably not threadsafe so maybe make the directory on GCP first!
    if not os.path.isdir(_OUTPUT_DIR.value):
        os.makedirs(_OUTPUT_DIR.value)

    logging.info(f"job details = nut={nut} n1={n1} n2={n2} n3={n3} lattice={lattice_type} atom_type={atom_type}")
    logging.info(f"output dir = {_OUTPUT_DIR.value}")
    n = [n1, n2, n3]
    run_nux_run(nut, atom_type, lattice_type, n)
    logging.info("DONE!")

if __name__ == "__main__":
    app.run(main)