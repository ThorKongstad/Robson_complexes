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
from ase.optimize import GPMin, BFGS
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from ase import Atoms
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from gpaw import GPAW, PW, Davidson
from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull
from gpaw.utilities import h2gpts
from gpaw.convergence_criteria import Density
from dftd4.ase import DFTD4


def optimiser_observer(atoms: Atoms, db_dir: str, db_id: int, goals: list[float]) -> None:
    global next_goal
    if next_goal != goals[0] and max([np.linalg.norm(force) for force in atoms.get_forces()]):
        next_goal = [goal for goal in goals if goal < next_goal][-1]
        if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms.copy(), relaxed=True, vibration=False, vib_en=False))


def main(db_id: int, db_dir: str, beta=0.05, maxold=5, weight=50, reset_spin: bool = False, diff_convergence: bool = False) -> None:

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        grid_spacing = row.get('grid_spacing')
        dftd4_bool = row.get('dftd4')
        charge = row.get('gpaw_charge')
#        dft_calc_pickle = eval(row.data.get('dft_calc_pickle'))
        initial_fmax = row.get('fmax')

    parprint(f'outstd of vib calculation with different calculator for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    hubberd_U = False
    hubberd_U_dict = dict(setups={'O': ':p,8.9,0', 'N': ':p,6.0,0', 'Fe': ':d,4.1,0', 'Co': ':d,4.4,0'}) if hubberd_U else {}

    if reset_spin:
        initial_magnetic_moments = [(3 if a.symbol == 'Co' else 4) * (-1 if i % 2 else 1) if a.symbol in ['Co', 'Fe'] else 0.0 for i, a in enumerate(atoms)]
        atoms.set_initial_magnetic_moments(initial_magnetic_moments)

    file_name = f'vib_id{db_id}_{structure_str}_{adsorbate_str}_diff_calc'

    if diff_convergence: convergence = {'convergence': {'density': Density(0.0005)}}
    else: convergence = {}

    calc_par_dict = dict(
        xc=functional,
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0, # or spinpol,
        mixer=MixerFull(beta=beta, nmaxold=maxold, weight=weight),
        charge=charge,
        txt=f'{functional_folder}/{file_name}.txt',
        symmetry='off',
        **hubberd_U_dict,
        **convergence
    )

    if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**calc_par_dict)])
    else: calc = GPAW(**calc_par_dict)
    atoms.set_calculator(calc)

    metal_symbol = ['Co','Fe']
    metal_at, not_metal_at = [], []
    for i, at in enumerate(atoms):
        if at.symbol in metal_symbol: metal_at.append(i)
        else: not_metal_at.append(i)
    metal_z_pos = list(pos[2] for pos in atoms[metal_at].get_positions())
    avg_metal_z_pos = np.mean(metal_z_pos)
    atoms_for_vib = list(filter(lambda i: (atoms[i].position[2] > (avg_metal_z_pos + 0.4)) or atoms[i].symbol in metal_symbol, list(range(len(atoms)))))
    locked_metals = list(filter(lambda i: (atoms[i].position[2] < (avg_metal_z_pos + 0.4)) and atoms[i].symbol not in metal_symbol, list(range(len(atoms)))))

    atoms.set_constraint(constraint=FixAtoms(locked_metals))
    atoms.get_forces()# fix incase it cant read forces, need to figure out a test for it. possible try TypeError or if self._cache['forces'] == None

    vib = Vibrations(atoms, indices=atoms_for_vib, name=f'{functional_folder}/{file_name}')
    vib.run()

    thermo = HarmonicThermo(vib.get_energies(), atoms.get_potential_energy(), ignore_imag_modes=True)

    if world.rank == 0:
        vib.summary(log=f'{functional_folder}/{file_name.replace("vib", "vib_en")}')

        with open(f'{functional_folder}/{file_name.replace("vib", "vib_en")}', 'r') as fil:
            energy_string = fil.read()

        # saving vib data
        update_db(db_dir, dict(id=db_id, vibration=True, zpe=thermo.get_ZPE_correction(), vib_en=energy_string,
                               enthalpy=thermo.get_internal_energy(300), entropy=thermo.get_entropy(300),
                               free_E=thermo.get_helmholtz_energy(300)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('--beta', '-b', default=0.05, type=float)
    parser.add_argument('--maxold','-mo', default=5, type=int)
    parser.add_argument('--weight', '-w', default=50, type=float)
    parser.add_argument('--reset_spin', '-r', default=False, action='store_true')
    parser.add_argument('--diff_con', '-dc', default=False, action='store_true')
    args = parser.parse_args()

    main(args.data_base_id, args.database, beta=args.beta, maxold=args.maxold, weight=args.weight, reset_spin=args.reset_spin, diff_convergence=args.diff_con)
