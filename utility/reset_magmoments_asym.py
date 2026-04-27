#partition=main
#nprocshared=16
#mem=2300MB

import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import ase.db as db
from ase.parallel import parprint, world, barrier


def main(db_id: int, db_dir: str, magmo_dict: dict[str, int]):

    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms = row.toatoms()

    initial_magnetic_moments = [magmo_dict[a.symbol] * (-1 if i % 2 else 1) if a.symbol in magmo_dict.keys() else 0.0 for i, a in enumerate(atoms)]
    atoms.set_initial_magnetic_moments(initial_magnetic_moments)

    if world.rank == 0: update_db(db_dir, dict(id=db_id, atoms=atoms, relaxed=False, vibration=False))


if __name__ == '__main__':
    class StrInt_Parser(argparse.Action):
        def __call__(self, parser, args, values: tuple[str, str], option_string=None):
            setattr(args, self.dest, (values[0], int(values[1])))


    class ParsePairs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) % 2 != 0:
                parser.error(f"Arguments must come in pairs (key value). Got {len(values)}.")

            result = {}
            for i in range(0, len(values), 2):
                key = values[i]
                try:
                    result[key] = int(values[i + 1])
                except ValueError:
                    parser.error(f"Value for '{key}' must be an integer, got '{values[i + 1]}'.")

            setattr(namespace, self.dest, result)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('atom_magmoment', help='"atom symbol" "magmoent" ... ', nargs='*', action=ParsePairs)
    args = parser.parse_args()

    main(args.data_base_id, args.database, args.atom_magmoment)