#partition=main
#nprocshared=16
#mem=2300MB

import argparse
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

from ase.optimize import GPMin
import ase.db as db
from ase.calculators.mixing import SumCalculator
from ase.parallel import parprint, world, barrier
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from dftd4.ase import DFTD4



def main(db_id: int, db_dir: str, fmax: float = 0.3):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        dftd4_bool = row.get('dftd4')
        dft_calc_pickle = eval(row.data.get('dft_calc_pickle'))

    parprint(f'outstd of opt calculation for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    functional_folder = sanitize(functional) + ('_D4' if dftd4_bool else '')
    if world.rank == 0: folder_exist(functional_folder)

    dft_calc_dict = pickle.loads(dft_calc_pickle)
    dft_calc_dict['txt'] = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}.txt'

    if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**dft_calc_dict)])
    else: calc = GPAW(**pickle.loads(dft_calc_pickle))

    atoms.set_calculator(calc)

    if isinstance(fmax, float) or isinstance(fmax, int): fmax = [fmax]
    for i, fm in enumerate(fmax):
        parprint(f'Starting {i+1} opt calculations with fmax={fm}')
        if isinstance(atoms.calc, SumCalculator):
            atoms.calc.mixer.calcs[1].txt = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}_{fm}fm.txt'
        else: atoms.calc.txt = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}_{fm}fm.txt'
        barrier()
        # define optimizer
        dyn = GPMin(atoms, trajectory=None)
        # run relaxation to a maximum force of 0.03 eV / Angstroms
        dyn.run(fmax=fm)
        if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=True, vibration=False, vib_en=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('--fmax', '-fm', default=0.3, type=float, nargs='+', help='state the fmax to use and if multiple runs the optimisation sequentially for each maximum force.')
    args = parser.parse_args()

    main(args.data_base_id, args.database)