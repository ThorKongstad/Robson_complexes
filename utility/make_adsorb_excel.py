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


def fill_gas_page(sheet):
    sheet.cell(2,1,'H2')
    sheet.cell(2,2,G_h2o)

    sheet.cell(3, 1, 'H20')
    sheet.cell(3, 2, G_h2o)

    sheet.cell(4, 1, 'O2')
    sheet.cell(4, 2, G_o2)

    sheet.cell(5,1,'U')
    sheet.cell(5, 1, 0)


def get_energy(panda, catalyst: str, charge: str, adsorbate: str, xc: str = 'RPBE', free_E_bool: bool = False) -> float:
    return panda.query('structure_str == @catalyst and gpaw_charge == @charge and xc == @xc and adsorbate_str == @adsorbate and energy.notna()').get('free_E' if free_E_bool else 'energy').iloc[0]


def main(db_directory: str, verbose: bool):
    pd_data = build_pd(db_directory)

    unique_catalysts = {struc for _, row in pd_data.iterrows() if not pd.isna((struc := row.get('structure_str')))}

    excel_file = xl.Workbook()
    gas_sheet = excel_file.active
    gas_sheet.title = 'gas molecules'

    fill_gas_page(gas_sheet)

    adsorbation_energy_sheet = excel_file.create_sheet('adsorption_E')
    adsorbation_free_sheet = excel_file.create_sheet('adsorption_G')

    for free_bool, sheet in zip((False, True), (adsorbation_energy_sheet, adsorbation_free_sheet)):
        sheet.cell(row=1, column=1).value = 'structure'
        sheet.cell(row=1, column=2).value = 'charge'
        sheet.cell(row=1, column=3).value = 'adsorbate'
        sheet.cell(row=1, column=4).value = 'RPBE'
        current_line = 2
        for i, struc in enumerate(unique_catalysts):
            unique_charges = {charg for _, row in pd_data.query('structure_str == @struc').iterrows() if not pd.isna((charg := row.get('gpaw_charge')))}
            for charge in unique_charges:
                unique_adsorbates = {struc for _, row in pd_data.query('structure_str == @struc and gpaw_charge == @charge').iterrows() if not pd.isna((struc := row.get('adsorbate_str'))) or struc != 'None'}
                sheet.cell(row=current_line, column=1).value = struc
                for j, adsorbate in enumerate(unique_adsorbates):
                    sheet.cell(row=current_line, column=3).value = adsorbate

                    match adsorbate:
                        case 'O':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - G_h2o + G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OH':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - G_h2o + 0.5 * G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OOH':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - 2 * G_h2o + 1.5 * G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OH_O':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - 2 * G_h2o + 1.5 * G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OH_OH':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - 2 * G_h2o + G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OOH_O':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - 3 * G_h2o + 2.5 * G_h2
                            except:
                                if verbose: traceback.print_exc()
                        case 'OOH_OH':
                            try:
                                sheet.cell(row=current_line, column=4).value = get_energy(pd_data, struc, charge, adsorbate, free_E_bool=free_bool) - get_energy(pd_data, struc, charge, 'None', free_E_bool=free_bool) - 3 * G_h2o + 2 * G_h2
                            except:
                                if verbose: traceback.print_exc()
                    current_line += 1

    excel_file.save('rmc_adsorp_dat.xlsx')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_directory', help='Path to the database')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args.db_directory,  args.verbose)
