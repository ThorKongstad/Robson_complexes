import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

from ase.constraints import FixAtoms, FixedPlane
from ase import Atoms
from ase.io import read, write


def main(atoms_dir: str, exclude_atoms: list[int], output: str):
    atoms = read(atoms_dir, index=-1)

    atoms.set_constraint(constraint=FixedPlane([i for i in range(len(atoms)) if i not in exclude_atoms], direction=[0, 0, 1]))

    write(output, atoms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('exclude_atoms', nargs='+', type=int)
    parser.add_argument('output')
    args = parser.parse_args()

    main(args.file, args.exclude_atoms, args.output)

