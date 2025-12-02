import argparse
import os.path
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import update_db
from ase.io import read
import ase.db as db
from ase import Atoms


def main(db_index: int, txt_dir: str, db_dir: str, relaxed: bool = False, new_constraint: bool = False, unsafe: bool = False):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    if not unsafe:
        with db.connect(db_dir) as db_obj:
            row = db_obj.get(selection=f'id={db_index}')
            functional = row.get('xc')
            old_atoms = row.toatoms()
            name = row.get('name')
            data = row.get('data')
    if not new_constraint and not unsafe:
        constraints = old_atoms.constraints

    if os.path.basename(txt_dir) == 'OUTCAR':
        from ase.calculators.vasp import Vasp
        updated_atoms: Atoms = read(txt_dir, index=-1, format='vasp-out')
        updated_atoms.calc = Vasp()
        freq, ifreq = updated_atoms.calc.read_vib_freq()
        zpe = sum(freq) / 1000
        if len(freq) > 0:
            data.update(dict(frequencies=freq, i_frequencies=ifreq))
            update_db(db_dir,
                      dict(
                          id=db_index,
                          atoms=updated_atoms,
                          relaxed=relaxed,
                          zpe=zpe,
                          vibration=True,
                          vib_en=False,
                          data=data
                      ))
            return
    elif os.path.basename(txt_dir) == 'CONTCAR': updated_atoms: Atoms = read(txt_dir, format='vasp')
    else: updated_atoms: Atoms = read(txt_dir, index=-1)

    if not new_constraint and not unsafe: updated_atoms.set_constraint(constraints)
    if not unsafe: assert old_atoms.get_chemical_formula() == updated_atoms.get_chemical_formula()
    update_db(db_dir,
              dict(
                  id=db_index,
                  atoms=updated_atoms,
                  relaxed=relaxed,
                  vibration=False,
                  vib_en=False
              ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_index', type=int)
    parser.add_argument('txt_directory', help='Path to the gpaw output.')
    parser.add_argument('db_directory', help='Path to the database')
    parser.add_argument('-relaxed', '--relaxed', action='store_true', default=False)
    parser.add_argument('-newCon', '--new_constraint', action='store_true', default=False)
    parser.add_argument('-unsafe', action='store_true', default=False)
    args = parser.parse_args()

    main(args.db_index, args.txt_directory, args.db_directory, args.relaxed, args.new_constraint, unsafe=args.unsafe)
