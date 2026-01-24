import argparse
import sys
import pathlib
from copy import copy
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, build_pd

import pandas as pd
import openpyxl as xl
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles.borders import Border, Side, BORDER_THIN


E_h2o = -13.926613
E_h2 = -6.727907

H2O_ts = 0.67
H2O_ZPE = 0.56
G_h2o = E_h2o + H2O_ZPE - H2O_ts

H2_ts = 0.41
H2_ZPE = 0.27
G_h2 = E_h2 + H2_ZPE - H2_ts

G_o2 = 2 * (2.46 + G_h2o - G_h2)

H2 = dict(
    Energy=-E_h2,
    zpe=H2_ZPE,
    entropy=H2_ts,
    free_E=G_h2o,
    solvation_E=0,
)

H2O = dict(
    Energy=E_h2o,
    zpe=H2O_ZPE,
    entropy=H2O_ts,
    free_E=G_h2o,
    solvation_E=0,
)

O2 = dict(
    Energy=2 * (2.46 + E_h2o - E_h2),
    zpe=0,
    entropy=0,
    free_E=G_o2,
    solvation_E=0,
)

def fill_gas_page(sheet):
    sheet.cell(2, 1, 'H2')
    sheet.cell(2, 2, G_h2o)

    sheet.cell(3, 1, 'H20')
    sheet.cell(3, 2, G_h2o)

    sheet.cell(4, 1, 'O2')
    sheet.cell(4, 2, G_o2)

#    sheet.cell(5, 1,'U')
#    sheet.cell(5, 2, 0)


def get_energy(panda, catalyst: str, charge: str, adsorbate: str, attribute: str, xc: str = 'RPBE') -> float:
    return panda.query('structure_str == @catalyst and gpaw_charge == @charge and xc == @xc and adsorbate_str == @adsorbate and energy.notna()').get(attribute).iloc[0]


def match_adsorbate(adsorbate: str, pd_data, struc: str, charge: str, attribute: str, xc: str, verbose: bool = False) -> float | None:
    match adsorbate:
        case 'O':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge, 'None', attribute, xc) - H2O[attribute] + H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OH':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge,'None', attribute, xc) - H2O[attribute] + 0.5 * H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OOH':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge,'None', attribute, xc) - 2 * H2O[attribute] + 1.5 * H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OH_O':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge, 'None', attribute, xc) - 2 * H2O[attribute] + 1.5 * H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OH_OH':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge, 'None', attribute, xc) - 2 * H2O[attribute] + H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OOH_O':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge, 'None', attribute, xc) - 3 * H2O[attribute] + 2.5 * H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case 'OOH_OH':
            try:
                return get_energy(pd_data, struc, charge, adsorbate, attribute, xc) - get_energy(pd_data, struc, charge, 'None', attribute, xc) - 3 * H2O[attribute] + 2 * H2[attribute]
            except:
                if verbose: traceback.print_exc()
        case _:
            if verbose: print(f'Adsorbate: {adsorbate} didnt match any of the cases')


def main(db_directory: str, verbose: bool):
    pd_data = build_pd(db_directory)

    unique_catalysts = {struc for _, row in pd_data.iterrows() if not pd.isna((struc := row.get('structure_str')))}
    unique_functional = {func for _, row in pd_data.iterrows() if not pd.isna((func := row.get('xc')))}

    excel_file = xl.Workbook()
    gas_sheet = excel_file.active
    gas_sheet.title = 'gas molecules'

    fill_gas_page(gas_sheet)

    upper_border = Border(top=Side(border_style=BORDER_THIN, color='00000000'))

    for sheet_name in unique_functional:
        working_sheet = excel_file.create_sheet(sheet_name)
        working_sheet.cell(row=1, column=1).value = 'structure'
        working_sheet.cell(row=1, column=2).value = 'charge'
        working_sheet.cell(row=1, column=3).value = 'adsorbate'
        working_sheet.cell(row=1, column=5).value = 'Energy'
        working_sheet.cell(row=1, column=6).value = 'zpe'
        working_sheet.cell(row=1, column=7).value = 'entropy'
        working_sheet.cell(row=1, column=8).value = 'free_E'

        working_sheet.cell(row=1, column=10).value = 'solvation'
        working_sheet.cell(row=1, column=11).value = 'total_free_E'
        for n in range(1, 12): working_sheet.cell(row=1, column=n).border = upper_border
        current_line = 2

        for i, struc in enumerate(unique_catalysts):
            unique_charges = {charg for _, row in pd_data.query('structure_str == @struc').iterrows() if not pd.isna((charg := row.get('gpaw_charge')))}
            working_sheet.cell(row=current_line, column=1).value = struc
            for charge in unique_charges:
                unique_adsorbates = {adsor for _, row in pd_data.query('structure_str == @struc and gpaw_charge == @charge and adsorbate_str != "None"').iterrows() if not pd.isna((adsor := row.get('adsorbate_str')))}
                working_sheet.cell(row=current_line, column=2).value = charge
                for j, adsorbate in enumerate(unique_adsorbates):
                    working_sheet.cell(row=current_line, column=3).value = adsorbate
                    for k, attri in enumerate(('energy', 'zpe', 'entropy', 'free_E')):
                        working_sheet.cell(row=current_line, column=5 + k).value = match_adsorbate(adsorbate, pd_data, struc, charge, attri, xc=sheet_name, verbose=verbose)

                    working_sheet.cell(row=current_line, column=10).value = match_adsorbate(adsorbate, pd_data, struc, charge, 'solvation_E', xc=sheet_name, verbose=verbose)
                    if working_sheet.cell(row=current_line, column=8).value is not None:
                        try:
                            working_sheet.cell(row=current_line, column=11).value = match_adsorbate(adsorbate, pd_data, struc, charge, 'free_E', xc=sheet_name, verbose=verbose) + match_adsorbate(adsorbate, pd_data, struc, charge, 'solvation_E', xc=sheet_name, verbose=verbose)
                        except:
                            if verbose: traceback.print_exc()
                    current_line += 1
            for cell_nr in range(1, 12): working_sheet.cell(row=current_line, column=cell_nr).border = upper_border
    excel_file.save('rmc_adsorp_dat.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_directory', help='Path to the database')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args.db_directory,  args.verbose)
