import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import ends_with, sanitize

from typing import Optional

from ase.io import read
from ase import Atoms
import ase.db as db
import pickle


def main(traj_structure: str, structure_str: str, functional_str: str, basis_str: str,  db_dir: str, charge: float = 0, spin: bool = False, adsorbate_str: Optional[str] = None, dftd4_bool: bool = False,):
    atoms: Atoms = read(traj_structure)

    if dftd4_bool: ends_with(functional_str, '-d4')

    calc_par_dict = dict(
        method=functional_str,
        basis=basis_str,
        spin=spin,
        charge=charge,
        output=f'{functional_str}_{sanitize(basis_str)}/{structure_str}_{adsorbate_str}'+'.txt',
    )

    calc_pickle = str(pickle.dumps(calc_par_dict))

    with db.connect(db_dir) as db_obj:
        db_obj.write(atoms=atoms, xc=functional_str, structure_str=structure_str, adsorbate_str=(adsorbate_str if adsorbate_str is not None else 'None'), relaxed=False, vibration=False, pyscf_charge=charge, dftd4=dftd4_bool, data=dict(dft_calc_pickle=calc_pickle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_traj')
    parser.add_argument('struc_str')
    parser.add_argument('functional',help='str denoting what functional to calculate with')
    parser.add_argument('basis', help='str denoting what basis to calculate with', default='gth-tzvp')
    parser.add_argument('db', help='name or directory for the database.')
    parser.add_argument('--adsorbate_str', '-ad')
    parser.add_argument('--charge', '-c', default=0, type=float)
    parser.add_argument('--spinpol', '-s', type=float, default=0)
    parser.add_argument('--dftd4', '-d4', action='store_true')

    args = parser.parse_args()

    main(traj_structure=args.struc_traj,
         structure_str=args.struc_str,
         functional_str=args.functional,
         basis_str=args.basis,
         adsorbate_str=args.adsorbate_str,
         db_dir=args.db,
         charge=args.charge,
         spin=args.spinpol,
         dftd4_bool=args.dftd4,
         )