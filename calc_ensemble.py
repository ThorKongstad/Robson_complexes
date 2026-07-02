#partition=main
#nprocshared=16
#mem=4000MB

import argparse
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import ase.db as db
from ase.dft.bee import BEEFEnsemble
from ase.calculators.mixing import SumCalculator
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from ase.parallel import parprint, world, barrier
from dftd4.ase import DFTD4


def main(db_id: int, db_dir: str = 'molreact.db'):

    # read from  database
    #atoms = read(f'/groups/kemi/thorkong/errors_investigation/molreact.db@id={db_id}')
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        if not row.get('relaxed'): raise ValueError(f"atoms at row id: {db_id} haven't been relaxed.")
        functional = row.get('xc')
#        if functional not in ('BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}"): raise ValueError(f'row {db_id}, is not a bee functional')
        atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        dftd4_bool = row.get('dftd4')
        dft_calc_pickle = eval(row.data.get('dft_calc_pickle'))
        if world.rank == 0:
            data_dict = row.get('data')

    parprint(f'outstd of ensemble calculation for db entry {db_id} with structure: {structure_str} and functional: {functional}')

    functional_folder = sanitize(functional)
    if world.rank == 0: folder_exist(functional_folder)

    if '{' in functional[0] and '}' in functional[-1] and ':' in functional: functional = eval(functional)

    barrier()
    dft_calc_dict = pickle.loads(dft_calc_pickle)

    dft_calc_dict['txt'] = f'{functional_folder}/opt_id{db_id}_{structure_str}_{adsorbate_str}.txt'
    if dftd4_bool:
        calc = SumCalculator([DFTD4(method=functional), GPAW(**dft_calc_dict)])
    else:
        calc = GPAW(**dft_calc_dict)
    atoms.set_calculator(calc)

    potential_e = atoms.get_potential_energy()
    ens = BEEFEnsemble(atoms)
    ensem_en_li = ens.get_ensemble_energies()

    if world.rank == 0:
        data_dict.update({'ensemble_en': ensem_en_li})
        update_db(db_dir, dict(id=db_id, ensemble_bool=True, data=data_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('database', help='directory to the database.')
    args = parser.parse_args()

    main(args.data_base_id, args.database)
