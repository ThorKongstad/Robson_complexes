#partition=main
#nprocshared=32
#mem=4000MB

import argparse
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import numpy as np
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from ase import Atoms
from ase.units import mol, kJ, kcal, Pascal, m
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull
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


def main(db_id: int, db_dir: str, fd_bool: bool = False, beta=0.05, maxold=5, weight=50):
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
        charge = row.get('gpaw_charge')
        grid_spacing = row.get('grid_spacing')

    parprint(f'outstd of solvation calculation with different calculator for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    hubberd_U = False
    hubberd_U_dict = dict(setups={'O': ':p,8.9,0', 'N': ':p,6.0,0', 'Fe': ':d,4.1,0', 'Co': ':d,4.4,0'}) if hubberd_U else {}

    dft_calc_dict = dict(
        xc=functional,
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0, # or spinpol,
        mixer=MixerFull(beta=beta, nmaxold=maxold, weight=weight),
        charge=charge,
        txt=f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}_diff_calc.txt',
        **hubberd_U_dict
    )

    if fd_bool:
        dft_calc_dict['mode'] = 'fd'

    dft_calc_dict['txt'] = f'{functional_folder}/sp{'_fd' if fd_bool else ''}_id{db_id}_{structure_str}_{adsorbate_str}.txt'
    calc = GPAW(**dft_calc_dict)
    atoms.set_calculator(calc)

    gas_E = atoms.get_potential_energy()

    atomic_radii = {'H': 1.09, 'C': 1.77, 'N': 1.66, 'O': 1.50, 'Co': 2.4, 'Fe': 2.44}

    dft_calc_dict['txt'] = f'{functional_folder}/solv{'_fd' if fd_bool else ''}_id{db_id}_{structure_str}_{adsorbate_str}.txt'

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
    parser.add_argument('-fd', '--finite_differences', action='store_true')
    parser.add_argument('--beta', '-b', default=0.05, type=float)
    parser.add_argument('--maxold', '-mo', default=5, type=int)
    parser.add_argument('--weight', '-w', default=50, type=float)
    args = parser.parse_args()

    main(args.data_base_id, args.database, args.finite_differences,args.beta, args.maxold, args.weight)
