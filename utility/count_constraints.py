
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, update_db

import ase.db as db
from ase import Atoms


def get_indices(constraint):
    try: return constraint.indices
    except AttributeError: return constraint.a


def main(db_id: int, db_dir: str, verbose: bool=False):
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        atoms: Atoms = row.toatoms()

    if len(atoms.constraints) > 0:
        print(f'Row number: {db_id} have {len(atoms.constraints)} constraints encompassing {sum(len(get_indices(con)) for con in atoms.constraints)} atoms.')
    else:
        if verbose:
            print(f'Row number: {db_id} has no constraints.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('database', help='directory to the database.')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args.data_base_id, args.database,verbose=args.verbose)
