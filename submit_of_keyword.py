import argparse
import ase.db as db
from ase.db.row import AtomsRow
import os
from subprocess import call
from typing import Sequence, Callable, Optional


def isgga(row: AtomsRow) -> bool: return row.get('xc') in ('PBE', 'RPBE', 'BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}")
def ismgga(row: AtomsRow) -> bool: return row.get('xc') in ('MGGA_X_REVM06_L+MGGA_C_REVM06_L', 'MGGA_X_TPSS+MGGA_C_TPSS', 'MGGA_X_R2SCAN+MGGA_C_R2SCAN', 'MGGA_X_R4SCAN+MGGA_C_R2SCAN', 'mBEEF-vdW')
def isbee(row: AtomsRow) -> bool: return row.get('xc') in ('BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}", 'mBEEF-vdW')
def coll_not_exist(row: AtomsRow, coll) -> bool: return row.get(coll) is None


def multi_filter_or(row: AtomsRow, funcs: Sequence[Callable[[AtomsRow], bool]]) -> bool: return any(func(row) for func in funcs)
def multi_filter_and(row: AtomsRow, funcs: Sequence[Callable[[AtomsRow], bool]]) -> bool: return all(func(row) for func in funcs)


def main(key: str, python_scribt: str, selection_filter: Optional[str] = None, db_dir: str = 'molreact.db', local: bool = False, slurm: str = '/groups/kemi/thorkong/katla_submission/submit_katla_GP228_static'):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    func_list = []
    if selection_filter is not None:
        for fil in selection_filter.split(','):
            temp_func_list = []
            for fi in fil.split('&&'):
                if fi == 'isgga': temp_func_list += [isgga]
                elif fi == 'ismgga': temp_func_list += [ismgga]
                elif fi == 'isbee': temp_func_list += [isbee]
                elif fi.split('=')[0] == 'coll_not_exist' or fil.split('=')[0].lower() == 'collnotexist': temp_func_list += [lambda x: coll_not_exist(x, fil.split('=')[-1])]
                else: print(f'{fil} for was not recognised as an implemented filter')
            if len(temp_func_list) != 0: func_list += [lambda x: multi_filter_and(x, temp_func_list.copy())]
    if len(func_list) != 0: selection_filter = {'filter': lambda x: multi_filter_or(x, func_list)}
    else: selection_filter = {}

    with db.connect(db_dir) as db_obj:
        row_iter = db_obj.select(selection=key, **selection_filter)

    for row in row_iter:
        if local: call(['python', python_scribt, str(row.get("id")), '-db', db_dir])
        else: call([slurm, python_scribt, str(row.get("id")), '-db', db_dir])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('keyword', help='must match keywords used in db selection')
    parser.add_argument('python_script')
    parser.add_argument('-db', '--database', help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    parser.add_argument('--filter', '-f', help='current implemented filters are isgga, ismgga and collNotExist="COLLOM" t. a "," denotes an or and "&&" denotes an and')
    parser.add_argument('--local', '-local', action='store_true')
    parser.add_argument('-ss', '--submission_script', help='directory to the slurm submission script.', default='/groups/kemi/thorkong/katla_submission/submit_katla_GP236_static')
    args = parser.parse_args()

    main(args.keyword, args.python_script, args.filter, args.database, local=args.local, slurm=args.submission_script)
