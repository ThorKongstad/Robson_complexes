import argparse
import os
import sys
import pathlib
import pickle
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import ase.db as db
from ase.parallel import parprint, world, barrier
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull, Davidson
from gpaw.mixer import FFTMixerFull, MixerDif, _definemixerfunc
from gpaw.utilities import h2gpts
from dftd4.ase import DFTD4


def main(db_id: int, db_dir: str, functional_str: str, grid_spacing: float = 0.16, charge: float = 0, spinpol: bool = False,  dftd4_bool: bool = False, hubberd_U: bool = False, reset_spin: bool = False, kpts: tuple[int,int,int] = (1, 1, 1), mixer: str = 'full', eigensolver: Optional[str] = None, beta=0.05, maxold=5, weight=50):
    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()
        structure_str = row.get('structure_str')
        adsorbate_str = row.get('adsorbate_str')
        if world.rank == 0:
            data_dict = row.get('data')

    match mixer:
        case 'full': mixer = MixerFull(beta=beta, nmaxold=maxold, weight=weight)
        case 'fFF': mixer = FFTMixerFull(beta=beta, nmaxold=maxold, weight=weight)
        case 'fb':
            BroydenMixerFull = _definemixerfunc('fullspin', 'broyden')
            mixer = BroydenMixerFull(beta=beta, nmaxold=maxold, weight=weight)
        case 'diff': mixer = MixerDif(beta=beta, nmaxold=maxold, weight=weight, beta_m=beta, nmaxold_m=1)
        case 'msr1': mixer = dict(backend='msr1', nmaxold=maxold, beta=beta)
        case _: raise NotImplementedError('Could not understand mixer')

    match eigensolver:
        case 'rmm': eigensolver_kwargs = dict(eigensolver=dict(name='rmm-diis'))
        case 'dav': eigensolver_kwargs = dict(eigensolver=Davidson(3))
        case 'cg': eigensolver_kwargs = dict(eigensolver=dict(name='cg'))
        case _: eigensolver_kwargs = dict()

    hubberd_U_dict = dict(setups={'O': ':p,8.9,0', 'N': ':p,6.0,0', 'Fe': ':d,4.1,0', 'Co': ':d,4.4,0'}) if hubberd_U else {}

    calc_par_dict = dict(
        xc=functional_str,
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        kpts=kpts,
        gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0 or spinpol,
        mixer=mixer,
        charge=charge,
        txt=f'{functional_str}/{structure_str}_{adsorbate_str}'+'.txt',
        **hubberd_U_dict
        **eigensolver_kwargs
    )
    calc_pickle = str(pickle.dumps(calc_par_dict))

    if world.rank == 0:
        data_dict.update(dict(dft_calc_pickle=calc_pickle))
        update_db(db_dir, dict(id=db_id, xc=functional_str, relaxed=False, vibration=False, grid_spacing=grid_spacing, gpaw_charge=charge, dftd4=dftd4_bool, data=data_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('functional',help='str denoting what functional to calculate with')
    parser.add_argument('--grid_spacing', '-g', type=float, default=0.16)
    parser.add_argument('--charge', '-c', default=0, type=float)
    parser.add_argument('--spinpol', '-s', action='store_true', help='sets spinpol to be true, spinpol will always be true if charge != 0')
    parser.add_argument('--dftd4', '-d4', action='store_true')
    parser.add_argument('--hubberd_U', '-HU', action='store_true')
    parser.add_argument('--reset_spin', '-rs', action='store_true')
    parser.add_argument('--kpts', '-k', nargs=3, type=float, default=(1, 1, 1))
    parser.add_argument('-eig', '--eigensolver', choices=['rmm', 'dav', 'cg'], default=None)
    parser.add_argument('-mix', '--mixer', choices=['full', 'fFF', 'fb', 'diff', 'msr1'], default='full')
    parser.add_argument('--beta', '-b', default=0.05, type=float)
    parser.add_argument('--maxold','-mo', default=5, type=int)
    parser.add_argument('--weight', '-w', default=50, type=float)
    args = parser.parse_args()

    main(args.data_base_id, args.database,
         functional_str=args.functional,
         grid_spacing=args.grid_spacing,
         charge=args.charge,
         spinpol=args.spinpol,
         dftd4_bool=args.dftd4,
         hubberd_U=args.hubberd_U,
         reset_spin=args.reset_spin,
         kpts=args.kpts,
         beta=args.beta, maxold=args.maxold, weight=args.weight,
         mixer=args.mixer,
         eigensolver=args.eigensolver,
         )
