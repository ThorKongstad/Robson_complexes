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

import ase
from ase import Atoms, Atom
from ase.io import read, write
from ase.calculators.mixing import SumCalculator
import numpy as np
from copy import deepcopy as dc
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import covalent_radii, atomic_numbers
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from gpaw import FermiDirac, PoissonSolver, Mixer, MixerFull
from dftd4.ase import DFTD4
from gpatom.beacon.beacon import (BEACON, SurrogateOptimizer,
                                   InitatomsGenerator, Checker)
from gpatom.gpfp.prior import RepulsivePotential, CalculatorPrior
from gpatom.beacon.str_gen import RandomBox, AtomsRelaxer, RandomMolecule, RandomStructure
from gpatom.gpfp.gp import GaussianProcess
from gpatom.gpfp.fingerprint import FingerPrint, FPUpdater
from gpatom.gpfp.avishart_hpfitter import HpFitterConstantRatioParallel
from gpatom.gpfp.atoms_gp_interface import LCBModel


def sph2cart(r, theta, phi):
    """Convert spherical to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2sph(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)             # polar angle
    phi = np.arctan2(y, x)               # azimuth
    return r, theta, phi


def cell_center_cartesian(atoms):
    """Return the Cartesian coordinates of the geometric center of the unit cell."""
    cell = atoms.get_cell()          # shape (3,3), cell vectors as rows
    frac_center = np.array([0.5, 0.5, 0.5])
    # fractional -> cartesian
    cart_center = frac_center.dot(cell)
    return cart_center


def rotation_matrix_z_to_vec(v):
    z = np.array([0.0, 0.0, 1.0])
    cos_a = np.clip(np.dot(z, v), -1, 1)
    if np.isclose(cos_a,  1.0): return np.eye(3)
    if np.isclose(cos_a, -1.0): return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, v)
    axis /= np.linalg.norm(axis)
    sin_a = np.sqrt(1.0 - cos_a**2)
    K = np.array([
        [ 0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


class OHGenerator:
    def __init__(self, rng, relaxer=None):
        self.rng = rng

    def get(self):

        atoms = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.96]])
#        atoms.constraints.append(FixBondLengths([[0,1]]))

        for axis in ['z']:
            angle = self.rng.uniform(0, 360)
            atoms.rotate(angle, axis, rotate_cell=False)
        return atoms


class OOHGenerator:
    def __init__(self, rng, relaxer=None):
        self.rng = rng

    def get(self):

        atoms = Atoms('OOH', positions=[[0, 0, 0], [0, 0, 1.330], [0.956, 0, 1.498]])
#        atoms.constraints.append(FixBondLengths([[0,1]]))
#        atoms.constraints.append(FixBondLengths([[1, 2]]))

        for axis in ['z']:
            angle = self.rng.uniform(0, 360)
            atoms.rotate(angle, axis, rotate_cell=False)
        return atoms


class AdsorbateRMC:
    """
    Generator for randomly placing molecules a structure at the center of the box.
    the generator doesn't change the molecule itself

    this functions is a modified copy of MoleculeOnSubstrate

    Examples:
    >>> sgen=MoleculeOnSubstrate(molecule_generator, adsorbate, dx=2)
    >>> new_atoms=sgen.get()
    """
    def __init__(self, molecule_generator, RMC_mol, binding_center: int,  dr: float = 3,
                 relaxer=None, **kwargs):

        """
        Parameters
        ----------
        molecule_generator: BEACON structure generator object
            A structure generator with a method molecule=molecule_generator.get()
            taking no inputs and outputting a molecule subject to custom
            generation logic, including setting of molecular constraints.
            MoleculeOnSubstrate knows how to handle molecules constrained by
            FixAtoms and FixBondLengths
            Required.

        substrate : ase.Atoms
            The substrate structure on which the adsorbate will be placed.
            Required.

        dr : float, optional
            Displacement interval in which the molecule can be placed from
            the center.
            Default is 3 Angstrom.

        relaxer : Relaxer object, optional
            Relaxation procedure applied after structure generation.
            Default is None. (No relaxation)

        **kwargs
            Additional keyword arguments passed to RandomStructure.
            May include:

            rng : random number generator, optional
                Random number generator for reproducibility.
                Default is `np.random` (not reproducible).
        """

        RandomStructure.__init__(self, RMC_mol, **kwargs)
        self.molecule_generator = molecule_generator
        self.RMC_mol: ase.Atoms = RMC_mol
        self.binding_center: int = binding_center
        self.dr = dr
        self.relaxer = relaxer

    def get(self):

        mol = self.molecule_generator.get()

        RMC_mol = self.RMC_mol.copy()
        mol.cell = RMC_mol.cell.copy()

        # pick directions for adsorbate
        adsorbate_orientation = self.random_unit_vector_in_cone()
        R = rotation_matrix_z_to_vec(adsorbate_orientation)
        mol.set_positions(mol.get_positions() @ R.T)

        radia_shift = self.rng.random() * self.dr + 1.05 * self.get_minimum_distance(adsorbate_orientation, RMC_mol[self.binding_center].position)

        mol.translate(RMC_mol[self.binding_center].position + radia_shift * adsorbate_orientation)

        atoms = RMC_mol+mol
        atoms.pbc = [False, False, False]

#        offset = len(RMC_mol)
#        constraints = []
#        for c in RMC_mol.constraints:
#            constraints.append(c)

#        for c in mol.constraints:
#            if isinstance(c, FixAtoms):
#                new_indices = [i + offset for i in c.index]
#                constraints.append(FixAtoms(indices=new_indices))
#            elif isinstance(c, FixBondLengths):
#                new_bonds = [[i + offset, j + offset] for (i, j) in c.pairs]
#                constraints.append(FixBondLengths(new_bonds))

#        atoms.set_constraint(constraints)

        if self.relaxer is not None:
            atoms = self.relaxer.run(atoms)

        return atoms

    def get_minimum_distance(self, O_center_unit_vector, center) -> float:
        min_distance = 0
        for at in self.RMC_mol:
            distance_from_at_O_vector = (center - at.position) - np.dot(np.dot((center - at.position), O_center_unit_vector), O_center_unit_vector)
            collision_size = covalent_radii[atomic_numbers['O']] + covalent_radii[atomic_numbers[at.symbol]]
            if np.linalg.norm(distance_from_at_O_vector) < collision_size:
                collision_distance_from_center = np.linalg.norm(np.dot(np.dot((center - at.position), O_center_unit_vector), O_center_unit_vector)) + np.sqrt(collision_size**2-np.linalg.norm(distance_from_at_O_vector)**2)
                if collision_distance_from_center > min_distance: min_distance = collision_distance_from_center
            else: continue
        return min_distance

    def random_unit_vector_in_cone(self, half_angle_deg: float = 45.0) -> np.ndarray:
        """
        Returns a random unit vector within a cone of given half-angle around the z-axis.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random generator (e.g. np.random.default_rng(seed))
        half_angle_deg : float
            Half-angle of the cone in degrees (default 45°)

        Returns
        -------
        np.ndarray
            Unit vector (x, y, z) of shape (3,)
        """
        half_angle_rad = np.deg2rad(half_angle_deg)

        # Sample cos(theta) uniformly in [cos(half_angle), 1]
        # This gives uniform density on the spherical cap
        cos_min = np.cos(half_angle_rad)
        cos_theta = self.rng.uniform(cos_min, 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)

        # Azimuthal angle uniform in [0, 2pi)
        phi = self.rng.uniform(0.0, 2.0 * np.pi)

        return np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])


def main(RMC_struture: str, RMC_binding_atom: int, adsorbate: str, charge: int):
    RMC_atoms: ase.Atoms = read(RMC_struture)

    rng = np.random.RandomState(42)

    match adsorbate:
        case 'OH': mol_gen = OHGenerator(rng=rng)
        case 'OOH': mol_gen = OOHGenerator(rng=rng)
        case _: raise NotImplementedError

    potential = RepulsivePotential(prefactor=10, rc=0.9)
    relaxer = AtomsRelaxer(calculator=potential, with_unit_cell=False)

    system_gen = AdsorbateRMC(mol_gen, RMC_atoms, RMC_binding_atom, dr=2, relaxer=relaxer, rng=rng)
    initatomsgen = InitatomsGenerator(sgen=system_gen, rgen=system_gen)

    prior = CalculatorPrior(potential, constant=0)

    gp_args = dict(prior=prior,
                   hp={'scale': 1000, 'weight': 100,
                       'ratio': 0.001, 'noisefactor': 1},
                   use_forces=True)

    gp = GaussianProcess(**gp_args)

    # next we set the size of a Atom fingerprint to  that of the largest atom
    # we will keep this as is since the size of the metals still matter and are so much larger than the adsorbate radii we are trying to add
    r_atom = max(covalent_radii[atomic_numbers[at.symbol]] for at in RMC_atoms)
    fp_args = {'r_cutoff': r_atom * 5, 'a_cutoff': r_atom * 3, 'aweight': 1}
    fp = FingerPrint(fp_args=fp_args, calc_strain=False)

    # optimiser for parameters
    hp_optimizer = HpFitterConstantRatioParallel()

    # fingerprint updater
    fp_updater = FPUpdater(factor=1 / 3)

    # finally everything is put together in the ML model
    model = LCBModel(gp=gp, fp=fp, hp_optimizer=hp_optimizer, fp_updater=fp_updater, kappa=2)

    surropt = SurrogateOptimizer(fmax=0.05, relax_steps=100,
                                 with_unit_cell=False)

    # the checker check structure likeness
    checker = Checker(dist_limit=0.5, rlimit=0.4)

    #folder = f'beacon_{os.path.basename(RMC_struture).split(".")[0]}_{adsorbate}'
    #folder_exist(folder)

    calc_par_dict = dict(
        xc='RPBE',
        basis='dzp',
        mode={'name': 'pw', 'ecut': 500, 'force_complex_dtype': True},
        gpts=h2gpts(0.16, RMC_atoms.get_cell(), idiv=4),
        parallel={'augment_grids': True, 'sl_auto': True},
        spinpol=charge != 0,  # or spinpol,
        mixer=MixerFull(beta=0.05, nmaxold=5, weight=50),
        charge=charge,
        txt=f'{os.path.basename(RMC_struture).split(".")[0]}_{adsorbate}.txt',
    )

    calculator = SumCalculator([DFTD4(method='RPBE'), GPAW(**calc_par_dict)])

    go = BEACON(calculator,
                model,
                initatomsgen,
                surropt=surropt,
                checker=checker,
                ninit=2,  # How many training points we want in initial set
                ndft=10,  # How many TBlite calculations are done in total
                nsur=3,  # How many surrogate optimizations are done per cycle
                write_surropt=True,  # Save surrogate relaxation end structures
                write_surropt_trajs=True)  # Dont save trajectory files for all surrogate relaxations
    # Run:
    go.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('RMC_struture')
    parser.add_argument('RMC_binding_atom', type=int)
    parser.add_argument('adsorbate')
    parser.add_argument('charge', type=int)

    main(**vars(parser.parse_args()))
