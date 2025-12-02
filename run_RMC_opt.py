#partition=main
#nprocshared=32
#mem=4000MB

import argparse
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import numpy as np
from ase.optimize import GPMin, BFGS
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from dftd4.ase import DFTD4


def optimiser_observer(atoms: Atoms, db_dir: str, db_id: int, goals: list[float]) -> None:
    global next_goal
    if next_goal != goals[0] and max([np.linalg.norm(force) for force in atoms.get_forces()]):
        next_goal = [goal for goal in goals if goal < next_goal][-1]
        if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms.copy(), relaxed=True, vibration=False, vib_en=False))


def main(db_id: int, db_dir: str, fmax: float = 0.3, restart: bool = False):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        dftd4_bool = row.get('dftd4')
        dft_calc_pickle = eval(row.data.get('dft_calc_pickle'))
        initial_fmax = row.get('fmax')

    parprint(f'outstd of opt calculation for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    dft_calc_dict = pickle.loads(dft_calc_pickle)

    dft_calc_dict['txt'] = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}.txt'
    if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**dft_calc_dict)])
    else: calc = GPAW(**pickle.loads(dft_calc_pickle))
    atoms.set_calculator(calc)

    if isinstance(fmax, float) or isinstance(fmax, int): fmax = [fmax]
    fmax = sorted(fmax)

    if len(fmax) > 1:
        global next_goal
        if initial_fmax:
            goals = [goal for goal in fmax if goal < initial_fmax and not restart]
        else: goals = fmax
        next_goal = fmax[-1]
    else: goals = fmax

    # define optimizer
    #dyn = GPMin(atoms, trajectory=None)
    dyn = BFGS(atoms, trajectory=None)
    if len(fmax) > 1: dyn.attach(optimiser_observer,
                                 atoms=atoms,
                                 db_dir=db_dir,
                                 db_id=db_id,
                                 goals=goals
                                 )
    barrier()
    # run relaxation to a maximum force of 0.03 eV / Angstroms
    dyn.run(fmax=goals[0])
    if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=True, vibration=False, vib_en=False))

#    for i, fm in enumerate(fmax):
#        if initial_fmax is not None and initial_fmax < fm and initial_fmax != 0: continue
#        parprint(f'Starting {i+1} opt calculations with fmax={fm}')
#        dft_calc_dict['txt'] = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}_{fm}fm.txt'
#        if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**dft_calc_dict)])
#        else: calc = GPAW(**pickle.loads(dft_calc_pickle))
#        atoms.set_calculator(calc)
#        barrier()
        # define optimizer
#        dyn = GPMin(atoms, trajectory=None)
        # run relaxation to a maximum force of 0.03 eV / Angstroms
#        dyn.run(fmax=fm)
#        if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=True, vibration=False, vib_en=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('--fmax', '-fm', default=0.3, type=float, nargs='+', help='state the fmax to use and if multiple runs the optimisation sequentially for each maximum force.')
    parser.add_argument('--restart', '-R')
    args = parser.parse_args()

    main(args.data_base_id, args.database, fmax=args.fmax, restart=args.restart)
