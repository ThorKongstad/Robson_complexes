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
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from ase import Atoms
from ase.units import mol, kJ, kcal, Pascal, m
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)


def optimiser_observer(atoms: Atoms, db_dir: str, db_id: int, goals: list[float]) -> None:
    global next_goal
    if next_goal != goals[0] and max([np.linalg.norm(force) for force in atoms.get_forces()]):
        next_goal = [goal for goal in goals if goal < next_goal][-1]
        if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms.copy(), relaxed=True, vibration=False, vib_en=False))


def main(db_id: int, db_dir: str):

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

    parprint(f'outstd of solvation calculation for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    dft_calc_dict = pickle.loads(dft_calc_pickle)

    dft_calc_dict['txt'] = f'{functional_folder}/sp_id{db_id}_{structure_str}_{adsorbate_str}.txt'
    calc = GPAW(**dft_calc_dict)
    atoms.set_calculator(calc)

    gas_E = atoms.get_potential_energy()

    atomic_radii = {'H': 1.09, 'C': 1.77, 'N': 1.66, 'O': 1.50, 'Co': 2.4, 'Fe': 2.44}

    dft_calc_dict['txt'] = f'{functional_folder}/solv_id{db_id}_{structure_str}_{adsorbate_str}.txt'

    calc = SolvationGPAW(
        cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii, 0.18),
        temperature=298.15,
        surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=78.36),
        interactions=[SurfaceInteraction(surface_tension=18.4*1e-3*Pascal*m)],
        **dft_calc_dict)
    atoms.set_calculator(calc)

    solvated_energy = atoms.get_potential_energy()

    barrier()
    # run relaxation to a maximum force of 0.03 eV / Angstroms
    if world.rank == 0: update_db(db_dir, dict(id=db_id, solvation_E=solvated_energy-gas_E))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    args = parser.parse_args()

    main(args.data_base_id, args.database)
