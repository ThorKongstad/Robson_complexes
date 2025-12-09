import argparse
#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
#from scripts_for_adsorbate_database import sanitize, folder_exist

from typing import Optional

from ase.io import read
from ase import Atoms
import ase.db as db
import pickle

from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull
from gpaw.utilities import h2gpts

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db


def main(db_id: int, db_dir: str,spinpol: bool = False,hubberd_U: bool = False):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()
        functional = row.get('xc')
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        dftd4_bool = row.get('dftd4')
        grid_spacing = row.get('grid_spacing')
        charge=row.get('gpaw_charge')
        data_dict = row.get('data')


    hubberd_U=False
    hubberd_U_dict = dict(
        setups={'O': ':p,8.9,0', 'N': ':p,6.0,0', 'Fe': ':d,4.1,0', 'Co': ':d,4.4,0'}) if hubberd_U else {}

    calc_par_dict = dict(
        xc=functional,
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0 or spinpol,
        mixer=MixerFull(beta=0.05, nmaxold=5, weight=50),
        charge=charge,
        txt=f'{functional}/{structure_str}_{adsorbate_str}' + '.txt',
        **hubberd_U_dict
    )

    calc_pickle = str(pickle.dumps(calc_par_dict))
    data_dict.update(dict(dft_calc_pickle=calc_pickle))

    update_db(db_dir, dict(id=db_id, data=data_dict))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('--spinpol', '-s', action='store_true', help='sets spinpol to be true, spinpol will always be true if charge != 0')
    parser.add_argument('--hubberd_U', '-HU', action='store_true')

    args = parser.parse_args()

    main(args.data_base_id, args.database)
