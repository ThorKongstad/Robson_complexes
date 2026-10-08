#partition=power
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
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.constraints import FixAtoms
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from dftd4.ase import DFTD4


def main(db_id: int, db_dir: str, unrestrict: bool = False, ideal: True = False):

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

    parprint(f'outstd of vib calculation for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    dft_calc_dict = pickle.loads(dft_calc_pickle)

    dft_calc_dict.update({'symmetry': 'off'})

    file_name = f'vib_id{db_id}_{structure_str}_{adsorbate_str}'
    dft_calc_dict['txt'] = f'{functional_folder}/{file_name}.txt'
    if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**dft_calc_dict)])
    else: calc = GPAW(**dft_calc_dict)
    atoms.calc = calc

    if not unrestrict:
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
    else:
        atoms_for_vib = list(range(len(atoms)))

    atoms.get_forces()# fix incase it cant read forces, need to figure out a test for it. possible try TypeError or if self._cache['forces'] == None

    vib = Vibrations(atoms, indices=atoms_for_vib, name=f'{functional_folder}/{file_name}')
    vib.run()

    vib_energies = vib.get_energies()
    if ideal:
        num_vibrating_atoms = len(atoms_for_vib)
        match ideal:
            case 'linear': n_modes = 3 * num_vibrating_atoms - 5
            case 'nonlinear': n_modes = 3 * num_vibrating_atoms - 6
            case 'monatomic': n_modes = 0
            case _: n_modes = len(vib_energies)
        n_modes = max(0, n_modes)

        sorted_energies = np.sort(vib_energies)
        filtered_vib_energies = sorted_energies[-n_modes:] if n_modes > 0 else np.array([])

        thermo = IdealGasThermo(filtered_vib_energies, potentialenergy=atoms.get_potential_energy(), geometry=ideal, atoms=atoms, ignore_imag_modes=True)
    else: thermo = HarmonicThermo(vib_energies, atoms.get_potential_energy(), ignore_imag_modes=True)

    if world.rank == 0:
        vib.summary(log=f'{functional_folder}/{file_name.replace("vib", "vib_en")}')

        with open(f'{functional_folder}/{file_name.replace("vib", "vib_en")}', 'r') as fil:
            energy_string = fil.read()

        # 1. Handle Enthalpy / Internal Energy discrepancy safely
        if hasattr(thermo, 'get_internal_energy'):
            internal_E = thermo.get_internal_energy(300)
        else:
            internal_E = thermo.get_enthalpy(300)

        # 2. Handle Free Energy discrepancy safely (Helmholtz vs Gibbs)
        if hasattr(thermo, 'get_helmholtz_energy'):
            free_energy = thermo.get_helmholtz_energy(300)
        else:
            # Ideal gas requires temperature and pressure (101325 Pa = 1 atm/bar standard)
            free_energy = thermo.get_gibbs_energy(temperature=300, pressure=101325.0)

        # saving vib data
        update_db(db_dir, dict(id=db_id, vibration=True, zpe=thermo.get_ZPE_correction(), vib_en=energy_string,
                               enthalpy=internal_E, entropy=thermo.get_entropy(300),
                               free_E=free_energy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--ideal', default=False, choices=['linear', 'nonlinear', 'monatomic'])
    args = parser.parse_args()

    main(args.data_base_id, args.database, unrestrict=args.full, ideal=args.ideal)
