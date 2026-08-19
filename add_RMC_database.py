import argparse
#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
#from scripts_for_adsorbate_database import sanitize, folder_exist

from typing import Optional

from ase.io import read
from ase import Atoms
import ase.db as db
import pickle

from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull, Davidson
from gpaw.mixer import FFTMixerFull, MixerDif, _definemixerfunc
from gpaw.utilities import h2gpts


def main(traj_structure: str, structure_str: str, functional_str: str,  db_dir: str, grid_spacing: float = 0.16, charge: float = 0, spinpol: bool = False, adsorbate_str: Optional[str] = None, dftd4_bool: bool = False, hubberd_U: bool = False, reset_spin: bool = False, mixer: str = 'full', eigensolver: Optional[str] = None, beta=0.05, maxold=5, weight=50):
    atoms: Atoms = read(traj_structure)

    if reset_spin:
        reset_spin_val = 3 if 'Co' in atoms.get_chemical_symbols() else 4

        initial_magnetic_moments = [reset_spin_val * (-1 if i % 2 else 1) if a.symbol in ['Co', 'Fe'] else 0.0 for i, a in enumerate(atoms)]
        atoms.set_initial_magnetic_moments(initial_magnetic_moments)

#    if charge != 0:
#        atoms.set_initial_charges()

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
        gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0 or spinpol,
        mixer=mixer,
        charge=charge,
        txt=f'{functional_str}/{structure_str}_{adsorbate_str}'+'.txt',
        **hubberd_U_dict,
        **eigensolver_kwargs
    )

    calc_pickle = str(pickle.dumps(calc_par_dict))

    with db.connect(db_dir) as db_obj:
        db_obj.write(atoms=atoms, xc=functional_str, structure_str=structure_str, adsorbate_str=(adsorbate_str if adsorbate_str is not None else 'None'), relaxed=False, vibration=False, grid_spacing=grid_spacing, gpaw_charge=charge, dftd4=dftd4_bool, data=dict(dft_calc_pickle=calc_pickle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_traj')
    parser.add_argument('struc_str')
    parser.add_argument('functional',help='str denoting what functional to calculate with')
    parser.add_argument('db',help='name or directory for the database.')
    parser.add_argument('--adsorbate_str', '-ad')
    parser.add_argument('--grid_spacing', '-g', type=float, default=0.16)
    parser.add_argument('--charge', '-c', default=0, type=float)
    parser.add_argument('--spinpol', '-s', action='store_true', help='sets spinpol to be true, spinpol will always be true if charge != 0')
    parser.add_argument('--dftd4', '-d4', action='store_true')
    parser.add_argument('--hubberd_U', '-HU', action='store_true')
    parser.add_argument('--reset_spin', '-rs', action='store_true')
    parser.add_argument('-eig', '--eigensolver', choices=['rmm', 'dav', 'cg'], default=None)
    parser.add_argument('-mix', '--mixer', choices=['full', 'fFF', 'fb', 'diff', 'msr1'], default='full')
    parser.add_argument('--beta', '-b', default=0.05, type=float)
    parser.add_argument('--maxold','-mo', default=5, type=int)
    parser.add_argument('--weight', '-w', default=50, type=float)
    args = parser.parse_args()

    main(traj_structure=args.struc_traj,
         structure_str=args.struc_str,
         functional_str=args.functional,
         adsorbate_str=args.adsorbate_str,
         db_dir=args.db,
         grid_spacing=args.grid_spacing,
         charge=args.charge,
         spinpol=args.spinpol,
         dftd4_bool=args.dftd4,
         hubberd_U=args.hubberd_U,
         reset_spin=args.reset_spin,
         beta=args.beta, maxold=args.maxold, weight=args.weight,
         mixer=args.mixer,
         eigensolver=args.eigensolver)
