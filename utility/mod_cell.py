#partition=main
#nprocshared=16
#mem=2300MB

import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import numpy as np
import ase.db as db
from ase.parallel import parprint, world, barrier


def main(db_id: int, db_dir: str, cell_mod: tuple[float, float, float]):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms = row.toatoms()

    new_cell = atoms.cell.copy()
    for i, mod in enumerate(cell_mod):
        vec = atoms.cell[i]
        length = np.linalg.norm(vec)
        if length == 0:
            raise ValueError(f"Cell vector {i} has zero length.")
        new_cell[i] = vec * (1.0 + mod / length)

    atoms.set_cell(new_cell, scale_atoms=False)

    if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=False, vibration=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('cell_mod', nargs=3, type=float)
    args = parser.parse_args()

    main(args.data_base_id, args.database, args.cell_mod)