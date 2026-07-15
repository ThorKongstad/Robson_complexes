#!/usr/bin/env python3
"""
Scan an ASE structure database for every unique (structure, charge, xc,
adsorbate) combination and collect each one into a dataclass with the same
fields as the lab's Excel report layout:

    structure str, adsorbate str, DFT+D4 E, ZPE, entropy, free energy,
    solvation E, Total G, adsorption no solv, adsorption, adsorption SD

'Total G' = free energy + solvation E.

'adsorption SD' is the 1-sigma uncertainty on the adsorption energy, derived
from each row's BEEF ensemble (2000 members). Only the resulting standard
deviation is kept -- the full ensembles are not retained. The same SD is
appropriate for both 'adsorption_no_solv' and 'adsorption', since ZPE,
entropy, and solvation are constant shifts and add no scatter of their own.

Adsorption energies (with and without solvation) are computed relative to
the matching bare structure (adsorbate_str == 'None') and the H2/H2O
gas-phase references from a separate molecule database, using the reaction
convention:
    slab + n_H2O H2O  --->  slab-adsorbate + n_H2 H2
For bare rows (adsorbate_str == 'None') there is nothing to reference against
themselves, so 'adsorption_no_solv' and 'adsorption' are left as None.

If a database contains more than one row for the same (structure, charge,
xc, adsorbate), the lowest-energy row is used (duplicates are typically
reruns/restarts of the same calculation).

Usage:
    python collect_adsorbates.py STRUCTURE_DB MOL_DB [--xc XC] [--verbose]
                                  [--xlsx OUTPUT.xlsx]

With --xlsx, an Excel workbook (openpyxl) is written alongside the stdout
table: sheet 1 "gas molecules" holds the H2/H2O reference data, sheet 2
"adsorbates" holds one row per AdsorbateResult, with 'structure str' written
only on the bare (None) row of each block and left blank below it. 'Total G'
and the adsorption energies are live Excel formulas; 'adsorption SD' is a
plain value (it comes from the BEEF ensemble, which isn't stored in the
sheet, so it can't be reconstructed by a formula).
"""
import argparse
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import ase.db as db
from ase.db.core import Database
from ase.db.row import AtomsRow
import openpyxl as xl
from openpyxl.worksheet.worksheet import Worksheet

FONT_NAME: str = 'Aptos Narrow'
ORR_O2_REF_EV: float = 4.92  # 4 * 1.23 eV: free energy of O2 + 4(H+ + e-) -> 2 H2O at U=0

# adsorbate_str -> (n_H2O, n_H2) such that
#   dE = E(ads) - E(bare) - n_H2O * E(H2O) + n_H2 * E(H2)
STOICHIOMETRY: Dict[str, Tuple[float, float]] = {
    'O':      (1, 1),
    'OH':     (1, 0.5),
    'OOH':    (2, 1.5),
    'OH_O':   (2, 1.5),
    'OH_OH':  (2, 1),
    'OOH_O':  (3, 2.5),
    'OOH_OH': (3, 2),
}


@dataclass
class AdsorbateResult:
    structure_str: str
    adsorbate_str: str
    charge: int
    xc: str
    energy: float                          # DFT+D4 E
    zpe: float
    entropy: float
    free_energy: float
    solvation_E: float
    total_G: float                         # = free_energy + solvation_E
    adsorption_no_solv: Optional[float]    # dE from 'energy' (electronic), no solvation. None if adsorbate_str == 'None'
    adsorption: Optional[float]            # dG from 'total_G' (free energy + solvation). None if adsorbate_str == 'None'
    adsorption_sd: Optional[float]         # 1-sigma uncertainty on 'adsorption', from the BEEF ensemble. None if adsorbate_str == 'None' or ensemble data unavailable


@dataclass
class PyTarget:
    """One block's Python-in-Excel limiting-potential formula anchor."""
    sheet: str
    cell: str            # e.g. 'R2' -- where the PY() formula goes
    data_range: str       # e.g. 'Q2:Q6' -- the U=0 range it reads
    cached_value: float


def combine(ads: float, bare: float, h2o: float, h2: float, n_h2o: float, n_h2: float) -> float:
    return ads - bare - n_h2o * h2o + n_h2 * h2


def ensemble_of(row: AtomsRow) -> npt.NDArray[np.float64]:
    try:
        return np.asarray(row.data['ensemble_en'])
    except (AttributeError, KeyError, TypeError):
        raise LookupError(f"row id={row.id} has no BEEF ensemble data ('ensemble_en')")


def adsorption_sd(target: AtomsRow, bare: AtomsRow, h2: AtomsRow, h2o: AtomsRow,
                   n_h2o: float, n_h2: float) -> Optional[float]:
    """1-sigma uncertainty on the adsorption energy, from the BEEF ensemble
    (2000 members per row). Only the resulting standard deviation is
    returned -- the ensembles themselves are not retained. Applies to both
    'adsorption_no_solv' and 'adsorption', since ZPE/entropy/solvation are
    constant shifts that don't add scatter of their own; this is why a
    single SD field covers both. Returns None if any of the four rows is
    missing ensemble data."""
    try:
        ens_t, ens_b = ensemble_of(target), ensemble_of(bare)
        ens_h2, ens_h2o = ensemble_of(h2), ensemble_of(h2o)
    except LookupError:
        return None
    n = min(len(ens_t), len(ens_b), len(ens_h2), len(ens_h2o))
    samples = combine(ens_t[:n], ens_b[:n], ens_h2o[:n], ens_h2[:n], n_h2o, n_h2)
    return float(np.std(samples))


def get_mol_row(mol_conn: Database, name: str, xc: str) -> AtomsRow:
    matches = list(mol_conn.select(structure_str=name, adsorbate_str='None', xc=xc))
    if not matches:
        raise LookupError(f"no reference row for '{name}' (xc='{xc}') found in molecule database")
    return min(matches, key=lambda r: r.energy)


def find_bare(conn: Database, structure_str: str, charge: int, xc: str) -> Optional[AtomsRow]:
    matches = list(conn.select(structure_str=structure_str, gpaw_charge=charge, xc=xc, adsorbate_str='None'))
    return min(matches, key=lambda r: r.energy) if matches else None


def collect_unique_rows(conn: Database) -> Dict[Tuple[str, int, str, str], AtomsRow]:
    """Group all rows by (structure, charge, xc, adsorbate) and keep the
    lowest-energy row in each group."""
    groups: Dict[Tuple[str, int, str, str], List[AtomsRow]] = {}
    for row in conn.select():
        structure = getattr(row, 'structure_str', None)
        adsorbate = getattr(row, 'adsorbate_str', None)
        charge = getattr(row, 'gpaw_charge', None)
        xc = getattr(row, 'xc', None)
        if None in (structure, adsorbate, charge, xc) or row.energy is None:
            continue
        groups.setdefault((structure, charge, xc, adsorbate), []).append(row)
    return {key: min(rows, key=lambda r: r.energy) for key, rows in groups.items()}


def collect_results(conn: Database, mol_conn: Database, verbose: bool = False) -> List[AdsorbateResult]:
    unique_rows = collect_unique_rows(conn)
    bare_cache: Dict[Tuple[str, int, str], Optional[AtomsRow]] = {}
    mol_cache: Dict[Tuple[str, str], AtomsRow] = {}

    results: List[AdsorbateResult] = []
    for (structure, charge, xc, adsorbate), row in sorted(unique_rows.items()):
        energy = float(row.energy)
        zpe = float(row.zpe)
        entropy = float(row.entropy)
        free_energy = float(row.free_E)
        solvation_e = float(row.solvation_E)
        total_g = free_energy + solvation_e

        adsorption_no_solv: Optional[float] = None
        adsorption: Optional[float] = None
        ads_sd: Optional[float] = None

        if adsorbate != 'None':
            bare_key = (structure, charge, xc)
            if bare_key not in bare_cache:
                bare_cache[bare_key] = find_bare(conn, structure, charge, xc)
            bare = bare_cache[bare_key]

            if bare is None:
                if verbose:
                    print(f"Warning: no bare reference for {structure} (charge={charge}, xc={xc}); "
                          f"leaving adsorption fields blank for adsorbate={adsorbate}", file=sys.stderr)
            elif adsorbate not in STOICHIOMETRY:
                if verbose:
                    print(f"Warning: unrecognized adsorbate '{adsorbate}' on {structure} "
                          f"(charge={charge}, xc={xc}); leaving adsorption fields blank", file=sys.stderr)
            else:
                n_h2o, n_h2 = STOICHIOMETRY[adsorbate]
                try:
                    for name in ('H2', 'H2O'):
                        if (name, xc) not in mol_cache:
                            mol_cache[(name, xc)] = get_mol_row(mol_conn, name, xc)
                    h2, h2o = mol_cache[('H2', xc)], mol_cache[('H2O', xc)]
                except LookupError as e:
                    if verbose:
                        print(f"Warning: {e}; leaving adsorption fields blank for "
                              f"{structure}/{adsorbate} (xc={xc})", file=sys.stderr)
                    h2 = h2o = None

                if h2 is not None and h2o is not None:
                    bare_energy = float(bare.energy)
                    bare_total_g = float(bare.free_E) + float(bare.solvation_E)
                    h2_total_g = float(h2.free_E) + float(h2.solvation_E)
                    h2o_total_g = float(h2o.free_E) + float(h2o.solvation_E)

                    adsorption_no_solv = combine(energy, bare_energy, float(h2o.energy), float(h2.energy),
                                                  n_h2o, n_h2)
                    adsorption = combine(total_g, bare_total_g, h2o_total_g, h2_total_g, n_h2o, n_h2)
                    ads_sd = adsorption_sd(row, bare, h2, h2o, n_h2o, n_h2)
                    if ads_sd is None and verbose:
                        print(f"Warning: no BEEF ensemble data for {structure}/{adsorbate} "
                              f"(charge={charge}, xc={xc}) or its references; adsorption_sd left blank",
                              file=sys.stderr)

        results.append(AdsorbateResult(
            structure_str=structure, adsorbate_str=adsorbate, charge=charge, xc=xc,
            energy=energy, zpe=zpe, entropy=entropy, free_energy=free_energy,
            solvation_E=solvation_e, total_G=total_g,
            adsorption_no_solv=adsorption_no_solv, adsorption=adsorption, adsorption_sd=ads_sd,
        ))
    return results


MOL_HEADERS: Tuple[str, ...] = (
    'name', 'energy', 'zpe', 'entropy', 'free energy', 'solvation E', 'Total G', 'energy SD')
ADS_HEADERS: Tuple[str, ...] = (
    'structure str', 'adsorbate str', 'charge', 'xc', 'DFT+D4 E', 'ZPE', 'entropy',
    'free energy', 'solvation E', 'Total  G', '', 'adsorption no solv', 'adsorption', 'adsorption SD',
    'orr state', 'state #', 'U=0', 'U=lim_pot')


def apply_font(ws: Worksheet) -> None:
    for row in ws.iter_rows():
        for c in row:
            if c.font is None or c.font.name != FONT_NAME:
                c.font = xl.styles.Font(name=FONT_NAME, bold=c.font.bold if c.font else False)


def style_header(ws: Worksheet, ncols: int) -> None:
    for col_idx in range(1, ncols + 1):
        c = ws.cell(row=1, column=col_idx)
        if c.value not in (None, ''):
            c.font = xl.styles.Font(name=FONT_NAME, bold=True)


def write_gas_molecules_sheet(wb: xl.Workbook, mol_conn: Database, xc: str) -> Dict[str, int]:
    """Writes H2 and H2O reference data. Returns {'H2': row, 'H2O': row}."""
    ws = wb.active
    ws.title = 'gas molecules'
    for i, h in enumerate(MOL_HEADERS, start=1):
        ws.cell(row=1, column=i, value=h)
    style_header(ws, len(MOL_HEADERS))

    row_of: Dict[str, int] = {}
    for r, name in enumerate(('H2', 'H2O'), start=2):
        row_obj = get_mol_row(mol_conn, name, xc)
        row_of[name] = r
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=float(row_obj.energy))
        ws.cell(row=r, column=3, value=float(row_obj.zpe))
        ws.cell(row=r, column=4, value=float(row_obj.entropy))
        ws.cell(row=r, column=5, value=float(row_obj.free_E))
        ws.cell(row=r, column=6, value=float(row_obj.solvation_E))
        ws.cell(row=r, column=7, value=f'=E{r}+F{r}')
        try:
            sd: Optional[float] = float(np.std(ensemble_of(row_obj)))
        except LookupError:
            sd = None
        ws.cell(row=r, column=8, value=sd)

    for col in 'ABCDEFGH':
        ws.column_dimensions[col].width = 14
    apply_font(ws)
    return row_of


def write_adsorbates_sheet(wb: xl.Workbook, results: List[AdsorbateResult],
                            mol_row_of: Dict[str, int]) -> List['PyTarget']:
    ws = wb.create_sheet('adsorbates')
    for i, h in enumerate(ADS_HEADERS, start=1):
        ws.cell(row=1, column=i, value=h)
    style_header(ws, len(ADS_HEADERS))

    h2_r, h2o_r = mol_row_of['H2'], mol_row_of['H2O']
    py_targets: List['PyTarget'] = []

    row = 2
    for (structure, charge, xc), group_iter in groupby(results, key=lambda r: (r.structure_str, r.charge, r.xc)):
        group = list(group_iter)
        row_of: Dict[str, int] = {}
        ads_g: Dict[str, Optional[float]] = {}

        for r in group:
            row_of[r.adsorbate_str] = row
            ads_g[r.adsorbate_str] = r.adsorption
            if r.adsorbate_str == 'None':
                ws.cell(row=row, column=1, value=r.structure_str)
            ws.cell(row=row, column=2, value=r.adsorbate_str)
            ws.cell(row=row, column=3, value=r.charge)
            ws.cell(row=row, column=4, value=r.xc)
            ws.cell(row=row, column=5, value=r.energy)
            ws.cell(row=row, column=6, value=r.zpe)
            ws.cell(row=row, column=7, value=r.entropy)
            ws.cell(row=row, column=8, value=r.free_energy)
            ws.cell(row=row, column=9, value=r.solvation_E)
            ws.cell(row=row, column=10, value=f'=H{row}+I{row}')

            if r.adsorption is not None and 'None' in row_of:
                n_h2o, n_h2 = STOICHIOMETRY[r.adsorbate_str]
                bare_r = row_of['None']
                ws.cell(row=row, column=12,
                        value=(f"=E{row}-E{bare_r}"
                               f"-'gas molecules'!$B${h2o_r}*{n_h2o:g}"
                               f"+'gas molecules'!$B${h2_r}*{n_h2:g}")).number_format = '0.00'
                ws.cell(row=row, column=13,
                        value=(f"=J{row}-J{bare_r}"
                               f"-'gas molecules'!$G${h2o_r}*{n_h2o:g}"
                               f"+'gas molecules'!$G${h2_r}*{n_h2:g}")).number_format = '0.00'
                if r.adsorption_sd is not None:
                    ws.cell(row=row, column=14, value=r.adsorption_sd).number_format = '0.00'
            row += 1

        # --- ORR pathway (O2 -> OOH -> O -> OH -> H2O), U=0, live formulas ---
        # Rows are assigned to states so that, top to bottom, they read state
        # 1..5 in order (needed for the PY() limiting-potential formula,
        # which treats the U=0 column as a plain sequential range). Each
        # state's value is a formula pointing at whichever data row actually
        # holds that adsorbate's own 'adsorption' figure.
        if all(a in row_of for a in ('None', 'O', 'OH', 'OOH')):
            block_row = row_of['None']
            state5_row = row  # one extra row below the block, for '*+H2O'
            ladder = (
                (block_row, '*+O2', 1, None),
                (row_of['O'], '*OOH', 2, f'=M{row_of["OOH"]}'),
                (row_of['OH'], '*O', 3, f'=M{row_of["O"]}'),
                (row_of['OOH'], '*OH', 4, f'=M{row_of["OH"]}'),
                (state5_row, '*+H2O', 5, None),
            )
            for r_row, label, state_n, formula in ladder:
                ws.cell(row=r_row, column=15, value=label)
                ws.cell(row=r_row, column=16, value=state_n)
                if formula is not None:
                    ws.cell(row=r_row, column=17, value=formula).number_format = '0.00'
                else:
                    ws.cell(row=r_row, column=17,
                            value=(ORR_O2_REF_EV if state_n == 1 else 0.0)).number_format = '0.00'
                # per-row energy under the (as yet unknown) limiting potential:
                # U=0 value shifted by state_n electrons' worth of the limiting potential
                if r_row != block_row:
                    ws.cell(row=r_row, column=18,
                            value=f'=Q{r_row}-$R${block_row}*(5-P{r_row})').number_format = '0.00'

            u0_states = [ORR_O2_REF_EV, ads_g['OOH'], ads_g['O'], ads_g['OH'], 0.0]
            lim_pot_value = _lim_pot(u0_states)
            p_cell = ws.cell(row=block_row, column=18, value=lim_pot_value)
            p_cell.number_format = '"U="0.00;"U=-"0.00'
            py_targets.append(PyTarget(sheet='adsorbates', cell=f'R{block_row}',
                                        data_range=f'Q{block_row}:Q{state5_row}', cached_value=lim_pot_value))
            row = state5_row + 1

    ws.column_dimensions['A'].width = 14
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['O'].width = 10
    apply_font(ws)
    return py_targets


def _lim_pot(states: List[float]) -> float:
    """Mirrors the PY()-formula logic: the consecutive step with the
    smallest |value| is the limiting potential."""
    diffs = [states[i] - states[i + 1] for i in range(len(states) - 1)]
    return min(diffs, key=lambda x: abs(x))


# --------------------------------------------------------------------------
# Python-in-Excel (PY()) injection
#
# openpyxl cannot write Excel's "Python in Excel" formulas, so this operates
# directly on the saved .xlsx (a zip of XML parts), following the exact
# structure Excel itself produces for a PY() cell (reverse-engineered from a
# workbook that already used one). The limiting-potential cell is written by
# openpyxl as a plain cached number first (so the value/format are already
# correct even without the formula), then this step swaps in the real
# formula. PY() requires Microsoft 365 "Python in Excel" (cloud execution) to
# recompute live; LibreOffice and older Excel cannot evaluate it.
# --------------------------------------------------------------------------

import re
import zipfile
from xml.sax.saxutils import escape

PYTHON_NS_RELTYPE = 'http://schemas.microsoft.com/office/2023/09/relationships/Python'
SHEETMETA_RELTYPE = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/sheetMetadata'

LIM_POT_SCRIPT = """
states=xl(%P2%, headers=False).squeeze()
lim_pot = min(map(lambda i: (states[i] - states[i+1]),range(len(states)-1)), key= lambda x: abs(x))
lim_pot
"""

PYTHON_ENV_INIT = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import excel
import warnings

warnings.simplefilter('ignore')

excel.set_xl_scalar_conversion(excel.convert_to_scalar)
excel.set_xl_array_conversion(excel.convert_to_dataframe)
"""


def _build_python_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<python xmlns="http://schemas.microsoft.com/office/spreadsheetml/2023/python">'
        '<environmentDefinition id="{9E9F4B6E-9C1D-4B7B-8B5B-6B6E9B7B9B7B}">'
        '<initialization userModified="1"><code xml:space="preserve">'
        f'{escape(PYTHON_ENV_INIT)}</code></initialization></environmentDefinition>'
        '<pythonScripts>'
        f'<pythonScript><code xml:space="preserve">{escape(LIM_POT_SCRIPT)}</code></pythonScript>'
        '</pythonScripts></python>'
    )


def _build_metadata_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<metadata xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:xda="http://schemas.microsoft.com/office/spreadsheetml/2017/dynamicarray">'
        '<metadataTypes count="1"><metadataType name="XLDAPR" minSupportedVersion="120000" '
        'copy="1" pasteAll="1" pasteValues="1" merge="1" splitFirst="1" rowColShift="1" '
        'clearFormats="1" clearComments="1" assign="1" coerce="1" cellMeta="1"/></metadataTypes>'
        '<futureMetadata name="XLDAPR" count="1"><bk><extLst>'
        '<ext uri="{bdbb8cdc-fa1e-496e-a857-3c3f30c029c3}">'
        '<xda:dynamicArrayProperties fDynamic="1" fCollapsed="0"/></ext></extLst></bk>'
        '</futureMetadata><cellMetadata count="1"><bk><rc t="1" v="0"/></bk></cellMetadata></metadata>'
    )


def _sheet_name_to_file(workbook_xml: str, rels_xml: str) -> Dict[str, str]:
    rid_to_name = dict(re.findall(r'<sheet name="([^"]*)"[^>]*r:id="(rId\d+)"', workbook_xml))
    rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels_xml))
    return {name: rid_to_target[rid] for name, rid in rid_to_name.items() if rid in rid_to_target}


def _next_rid(rels_xml: str) -> int:
    nums = [int(n) for n in re.findall(r'Id="rId(\d+)"', rels_xml)]
    return (max(nums) + 1) if nums else 1


def _patch_cell(sheet_xml: str, ref: str, data_range: str, cached_value: float) -> str:
    pattern = re.compile(rf'<c r="{re.escape(ref)}"([^>]*)>.*?</c>')
    m = pattern.search(sheet_xml)
    if not m:
        raise LookupError(f"could not find placeholder cell {ref} to patch")
    attrs = m.group(1)
    if 'cm=' not in attrs:
        attrs += ' cm="1"'
    formula = f'_xlfn._xlws.PY(0,0,{data_range})'
    new_cell = f'<c r="{ref}"{attrs}><f t="array" ref="{ref}">{escape(formula)}</f><v>{cached_value!r}</v></c>'
    return sheet_xml[:m.start()] + new_cell + sheet_xml[m.end():]


def inject_python_formulas(xlsx_path: str, py_targets: List[PyTarget]) -> None:
    with zipfile.ZipFile(xlsx_path, 'r') as zin:
        parts = {name: zin.read(name) for name in zin.namelist()}

    content_types = parts['[Content_Types].xml'].decode('utf-8')
    rels = parts['xl/_rels/workbook.xml.rels'].decode('utf-8')
    workbook_xml = parts['xl/workbook.xml'].decode('utf-8')
    sheet_files = _sheet_name_to_file(workbook_xml, rels)

    parts['xl/python.xml'] = _build_python_xml().encode('utf-8')
    parts['xl/metadata.xml'] = _build_metadata_xml().encode('utf-8')

    if '/xl/python.xml' not in content_types:
        content_types = content_types.replace(
            '</Types>',
            '<Override PartName="/xl/python.xml" ContentType="application/vnd.ms-excel.python+xml"/></Types>')
    if '/xl/metadata.xml' not in content_types:
        content_types = content_types.replace(
            '</Types>',
            '<Override PartName="/xl/metadata.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheetMetadata+xml"/>'
            '</Types>')
    parts['[Content_Types].xml'] = content_types.encode('utf-8')

    next_id = _next_rid(rels)
    additions = ''
    if PYTHON_NS_RELTYPE not in rels:
        additions += f'<Relationship Id="rId{next_id}" Type="{PYTHON_NS_RELTYPE}" Target="python.xml"/>'
        next_id += 1
    if SHEETMETA_RELTYPE not in rels:
        additions += f'<Relationship Id="rId{next_id}" Type="{SHEETMETA_RELTYPE}" Target="metadata.xml"/>'
        next_id += 1
    if additions:
        rels = rels.replace('</Relationships>', additions + '</Relationships>')
    parts['xl/_rels/workbook.xml.rels'] = rels.encode('utf-8')

    by_sheet: Dict[str, List[PyTarget]] = {}
    for t in py_targets:
        by_sheet.setdefault(t.sheet, []).append(t)
    for sheet_name, targets in by_sheet.items():
        sheet_file = f'xl/worksheets/{sheet_files[sheet_name]}'
        xml_text = parts[sheet_file].decode('utf-8')
        for t in targets:
            xml_text = _patch_cell(xml_text, t.cell, t.data_range, t.cached_value)
        parts[sheet_file] = xml_text.encode('utf-8')

    with zipfile.ZipFile(xlsx_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in parts.items():
            zout.writestr(name, data)


def write_excel(results: List[AdsorbateResult], mol_conn: Database, output: str, xc: str) -> None:
    wb = xl.Workbook()
    mol_row_of = write_gas_molecules_sheet(wb, mol_conn, xc)
    py_targets = write_adsorbates_sheet(wb, results, mol_row_of)
    wb.save(output)
    if py_targets:
        inject_python_formulas(output, py_targets)


def main(struc_db: str, mol_db: str, xc_filter: Optional[str] = None, verbose: bool = False,
         xlsx_output: Optional[str] = None) -> None:
    conn = db.connect(struc_db)
    mol_conn = db.connect(mol_db)

    results = collect_results(conn, mol_conn, verbose=verbose)
    if xc_filter:
        results = [r for r in results if r.xc == xc_filter]

    if not results:
        sys.exit(f"Error: no rows found in {struc_db}" + (f" for xc={xc_filter}" if xc_filter else ""))

    header = (f"{'structure':<16}{'adsorbate':<10}{'charge':>7}{'xc':>7}{'DFT+D4 E':>12}"
              f"{'ZPE':>8}{'entropy':>10}{'free E':>12}{'solv E':>10}{'Total G':>12}"
              f"{'ads (no solv)':>16}{'adsorption':>13}{'ads SD':>10}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r.structure_str:<16}{r.adsorbate_str:<10}{r.charge:>7}{r.xc:>7}{r.energy:>12.4f}"
              f"{r.zpe:>8.4f}{r.entropy:>10.6f}{r.free_energy:>12.4f}{r.solvation_E:>10.4f}{r.total_G:>12.4f}"
              f"{'' if r.adsorption_no_solv is None else f'{r.adsorption_no_solv:.4f}':>16}"
              f"{'' if r.adsorption is None else f'{r.adsorption:.4f}':>13}"
              f"{'' if r.adsorption_sd is None else f'{r.adsorption_sd:.4f}':>10}")
    print(f"\n{len(results)} unique (structure, charge, xc, adsorbate) row(s)")

    if xlsx_output:
        xcs = sorted({r.xc for r in results})
        if len(xcs) > 1:
            sys.exit(f"Error: multiple xc functionals found ({', '.join(xcs)}); "
                      f"pass --xc to pick one (the 'gas molecules' sheet holds a single H2/H2O reference).")
        write_excel(results, mol_conn, xlsx_output, xcs[0])
        print(f"Wrote {xlsx_output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('struc_db', help='Path to the ASE database with the slab/adsorbate structures')
    parser.add_argument('mol_db', help='Path to the ASE database with the gas-phase references H2/H2O')
    parser.add_argument('--xc', default=None, help='Restrict output to a single xc functional')
    parser.add_argument('--xlsx', default=None, metavar='OUTPUT.xlsx',
                         help='Also write an Excel workbook (gas molecules + adsorbates sheets) to this path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args.struc_db, args.mol_db, xc_filter=args.xc, verbose=args.verbose, xlsx_output=args.xlsx)