import os
import time
from typing import NoReturn, Sequence, Tuple, Never, Optional, NamedTuple
from dataclasses import dataclass, field
from itertools import chain
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import numpy as np
import ase.db as db
import pandas as pd
from sqlite3 import OperationalError


@dataclass
class reaction:
    reactants: Sequence[Tuple[str, float]]
    products: Sequence[Tuple[str, float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n:.2g}{smi if smi != "cid281" else "C|||O"}' for smi, n in self.reactants]) + ' ---> ' + ' + '.join([f'{n:.2g}{smi if smi != "cid281" else "C|||O"}' for smi, n in self.products])


class component(NamedTuple):
    type: str
    name: str
    amount: float


@dataclass
class adsorbate_reaction:
    reactants: Tuple[Tuple[str, str, float] | component, ...]
    products: Tuple[Tuple[str, str, float] | component, ...]

    def __post_init__(self):
        #component = namedtuple('component', ['type', 'name', 'amount'])
        for n, reacs_or_prods in enumerate([self.reactants, self.products]):
            new_component_seq = []
            for i, reac_or_prod in enumerate(reacs_or_prods):
                if len(reac_or_prod) != 3: raise ValueError('a component of a reaction does not have the correct size')
                if not reac_or_prod[0] in ('molecule', 'slab', 'adsorbate'): raise ValueError('The reactant or product type string appear to be wrong')
                new_component_seq.append(component(*reac_or_prod) if not isinstance(reac_or_prod, component) else reac_or_prod)
            setattr(self, 'reactants' if n == 0 else 'products', tuple(new_component_seq))

    def __str__(self):
        return ' ---> '.join([' + '.join([f'{reac.amount:.2g}{reac.name if reac.name != "cid281" else "C|||O"}({reac.type})' for reac in comp]) for comp in (self.reactants,self.products)])


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'", '"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|', ' ', ',', '.']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':', ';']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


#def folder_exist(folder_name: str, path: str = '.', tries: int = 10) -> NoReturn:
#    try:
#        tries -= 1
#        if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/') + folder_name)
#    except FileExistsError:
#        time.sleep(2)
#        if tries > 0: folder_exist(folder_name, path=path, tries=tries)


@retry(retry=retry_if_exception_type(FileExistsError), stop=stop_after_attempt(5), wait=wait_fixed(2))
def folder_exist(folder_name: str, path: str = '.') -> None:
    if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/') + folder_name)


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


def overpotential(dG_OOH: float, dG_OH: float, dG_O: float) -> float: return min((4.92 - dG_OOH, dG_OOH - dG_O, dG_O - dG_OH, dG_OH)) # 1.23 -


def mean(values: Sequence[float]) -> float: return sum(values) / len(values)


def sd(values: Sequence[float], mean_value: Optional[float] = None) -> float:
    if not mean_value: mean_value = mean(values)
    return np.sqrt((1 / len(values)) * sum(((x - mean_value) ** 2 for x in values)))


@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(5), wait=wait_fixed(10))
def update_db(db_dir: str, db_update_args: dict):
    with db.connect(db_dir) as db_obj:
        db_obj.update(**db_update_args)


adsorption_OH_reactions = tuple(chain(*((
        adsorbate_reaction((('molecule', 'O=O', 0.5), ('molecule', '[HH]', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1),)),
        adsorbate_reaction((('molecule', 'O', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1), ('molecule', '[HH]', 0.5))),
        adsorbate_reaction((('molecule', 'OO', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1),)),
                        )for metal in ['Pt', 'Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))

adsorption_OOH_reactions = tuple(chain(*((
        adsorbate_reaction((('molecule', 'O=O', 1), ('molecule', '[HH]', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1),)),
        adsorbate_reaction((('molecule', 'O', 2), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('molecule', '[HH]', 1.5))),
        adsorbate_reaction((('molecule', 'OO', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('molecule', '[HH]', 0.5))),
                        )for metal in ['Pt', 'Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))


adsorption_O_reactions = tuple(chain(*((
        adsorbate_reaction((('molecule', 'O=O', 1/2), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_fcc', 1),)),
        adsorbate_reaction((('molecule', 'O', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_fcc', 1), ('molecule', '[HH]', 1))),
        adsorbate_reaction((('molecule', 'OO', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_fcc', 1), ('molecule', '[HH]', 0.5))),
                        )for metal in ['Pt', 'Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))

adsorption_O_reactions_top = tuple(chain(*((
        adsorbate_reaction((('molecule', 'O=O', 1/2), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_top', 1),)),
        adsorbate_reaction((('molecule', 'O', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_top', 1), ('molecule', '[HH]', 1))),
        adsorbate_reaction((('molecule', 'OO', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_top', 1), ('molecule', '[HH]', 0.5))),
                        )for metal in ['Pt', 'Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))

metal_ref_ractions = tuple(chain(*((
        adsorbate_reaction((('adsorbate', 'Pt_111_OH_top', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1), ('slab', 'Pt_111', 1))),
        adsorbate_reaction((('adsorbate', 'Pt_111_OOH_top', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('slab', 'Pt_111', 1))),
        adsorbate_reaction((('adsorbate', 'Pt_111_O_fcc', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_O_fcc', 1), ('slab', 'Pt_111', 1))),
                        )for metal in ['Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))


metal_OOH_OH_ref_ractions = tuple(chain(*((
        adsorbate_reaction((('adsorbate', 'Pt_111_OH_top', 1), ('slab', f'{metal}_111', 1), ('molecule', 'O', 1),), (('adsorbate', f'{metal}_111_OOH_top', 1), ('slab', 'Pt_111', 1), ('molecule', '[HH]', 1))),
                        )for metal in ['Cu', 'Pd', 'Rh', 'Ag', 'Ir', 'Au'])))


all_adsorption_reactions = adsorption_OH_reactions + adsorption_OOH_reactions + metal_ref_ractions


def build_pd(db_dir_list, select_key: Optional = None):
    if isinstance(db_dir_list, str): db_dir_list = [db_dir_list]
    for db_dir in db_dir_list:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'):
            raise FileNotFoundError("Can't find database")
    db_list = [db.connect(work_db) for work_db in db_dir_list]
    pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select(selection=select_key)])
    return pd_dat


__all__ = [sanitize, folder_exist, ends_with, update_db, reaction, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions, all_adsorption_reactions, adsorption_O_reactions, adsorption_O_reactions_top, mean, sd, overpotential, metal_OOH_OH_ref_ractions]

