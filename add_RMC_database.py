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


def main(traj_structure:str, structure_str: str, functional_str: str,  db_dir: str, grid_spacing: float = 0.16, charge: float = 0, spinpol: bool = False, adsorbate_str: Optional[str] = None, dftd4_bool: bool = False ):
    atoms: Atoms = read(traj_structure)


    calc_par_dict = dict(
        xc=functional_str,
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        gpts=h2gpts(0.16, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0 or spinpol,
        mixer=MixerFull(beta=0.05, nmaxold=5, weight=50),
        charge=charge,
        setups={'O': ':p,8.9,0', 'N': ':p,6.0,0', 'Fe': ':d,4.1,0', 'Co': ':d,4.4,0'},
        txt=f'{functional_str}/{structure_str}'+'.txt',
    )

    #atoms.set_calculator(calc_par_dict)

    calc_pickle = str(pickle.dumps(calc_par_dict))

    with db.connect(db_dir) as db_obj:
        db_obj.write(atoms=atoms, xc=functional_str, structure_str=structure_str, adsorbate_str=(adsorbate_str if adsorbate_str is not None else 'NaN'), relaxed=False, vibration=False, grid_spacing=grid_spacing, dftd4=dftd4_bool, data=dict(dft_calc_pickle=calc_pickle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_traj')
    parser.add_argument('struc_str')
    parser.add_argument('functional',help='str denoting what functional to calculate with')
    parser.add_argument('db',help='name or directory for the database.')
    parser.add_argument('--adsorbate_str', '-ad')
    parser.add_argument('--grid_spacing', '-g', default=0.16)
    parser.add_argument('--charge', '-c', default=0)
    parser.add_argument('--spinpol', '-s', action='store_true', help='sets spinpol to be true, spinpol will always be true if charge != 0')
    parser.add_argument('--dftd4', '-d4', action='store_true')
    args = parser.parse_args()

    main(traj_structure=args.struc_traj, structure_str=args.struc_str, functional_str=args.functional, adsorbate_str=args.adsorbate_str, db_dir=args.db, grid_spacing=args.grid_spacing, charge=args.charge, spinpol=args.spinpol, dftd4_bool=args.dft4)
