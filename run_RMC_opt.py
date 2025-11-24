#partition=katla
#nprocshared=32
#mem=2300MB
#constrain='[v1|v2|v3|v4|v5]'

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



def main(db_id: int, db_dir: str = 'molreact.db'):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        grid_spacing= row.get('grid_spacing')
        dftd4_bool = row.get('dftd4')
        dft_calc_pickle = eval(row.data.get('dft_calc_pickle'))

    parprint(f'outstd of opt calculation for db entry {db_id} with structure: {structure_str}, adsorbate: {adsorbate_str} and functional: {functional}')

    if not grid_spacing:
        grid_spacing = 0.16
        parprint('grid spacing could not be found in the database entry and was set to 0.16')

    functional_folder = sanitize(functional)
    if world.rank == 0: folder_exist(functional_folder)

    if dftd4_bool: calc = SumCalculator([DFTD4(method=functional), GPAW(**pickle.loads(dft_calc_pickle))])
    else: calc = GPAW(**pickle.loads(dft_calc_pickle))

    atoms.set_calculator(calc)

    barrier()

    # define optimizer
    dyn = GPMin(atoms, trajectory=None)
    # run relaxation to a maximum force of 0.03 eV / Angstroms
    dyn.run(fmax=0.03)
    if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=True, vibration=False, vib_en=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, args.database)