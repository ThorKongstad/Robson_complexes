import argparse
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import numpy as np
from ase.constraints import FixAtoms, FixedPlane
import ase.db as db
from ase import Atoms


def main(db_id: int, db_dir: str):
    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()

    metal_symbol = ['Co','Fe']
    metal_at, not_metal_at = [], []
    for i, at in enumerate(atoms):
        if at.symbol in metal_symbol: metal_at.append(i)
        else: not_metal_at.append(i)
    metal_z_pos = list(pos[2] for pos in atoms[metal_at].get_positions())
    avg_metal_z_pos = np.mean(metal_z_pos)
    atoms_for_vib = list(filter(lambda i: (atoms[i].position[2] > (avg_metal_z_pos + 0.4)) or atoms[i].symbol in metal_symbol, list(range(len(atoms)))))
    locked_metals = list(filter(lambda i: (atoms[i].position[2] < (avg_metal_z_pos + 0.4)) and atoms[i].symbol not in metal_symbol, list(range(len(atoms)))))

    atoms.set_constraint(constraint=FixedPlane(locked_metals, direction=[0, 0, 1]))

    update_db(db_id, atoms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    args = parser.parse_args()

    main(args.data_base_id, args.database)

