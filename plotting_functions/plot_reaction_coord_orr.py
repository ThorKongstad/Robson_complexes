import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback
from re import match

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, build_pd
#from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot(): pass


def main(db_dir: list[str|pathlib.Path]):
    pd_catalysts = build_pd(db_dir)

    unique_catalysts = {struc for _, row in pd_catalysts.iterrows() if not pd.isna((struc := row.get('structure_str')))}
    unique_charges = {charg for _, row in pd_catalysts.iterrows() if not pd.isna((charg := row.get('charge')))}
    unique_functionals = {xc for _, row in pd_catalysts.iterrows() if not pd.isna((xc := row.get('xc')))}

    # Small molecule energies
    E_h2o = -13.926613
    E_h2 = -6.727907

    H2O_ts = 0.67
    H2O_ZPE = 0.56
    G_h2o = E_h2o + H2O_ZPE - H2O_ts

    H2_ts = 0.41
    H2_ZPE = 0.27
    G_h2 = E_h2 + H2_ZPE - H2_ts

    G_o2 = 2 * (2.46 + G_h2o - G_h2)

    U = 1.23

    fig = go.Figure()
    for xc in unique_charges:
        for cat in unique_catalysts:
            for charge in unique_charges:
                try:
                    dft_E_None = pd_catalysts.query('structure_str == @xc and charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "None"').get('energy').iloc[0]
                    dft_E_O = pd_catalysts.query('structure_str == @xc and charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "O"').get('energy').iloc[0]
                    dft_E_OH = pd_catalysts.query('structure_str == @xc and charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "OH"').get('energy').iloc[0]
                    dft_E_OOH = pd_catalysts.query('structure_str == @xc and charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "OOH"').get('energy').iloc[0]

                    state_1 = dft_E_None + G_o2 + 4*G_h2/2
                    state_2 = dft_E_OOH + 3*G_h2/2 - 0.3
                    state_3 = dft_E_O + 2*G_h2/2 + G_h2o
                    state_3_alt = dft_E_O + 2*G_h2/2
                    state_4 = dft_E_OH + G_h2/2 + G_h2o - 0.3
                    state_5 = dft_E_None + 2*G_h2o

                    fig.add_trace(go.Scatter(
                        mode='lines',
                        name=f'{xc}-{cat}-charge:{charge}',
                        x=('O2', '*OOH', '*O', '*OH', '*'),
                        y=(4.92-4*U, state_2-state_1-4.92-3*U, state_3-state_1-4.92-2*U, state_4-state_1-4.92-U, 0),
                        legend=cat,
                        legendgroup=cat,
                    ))
                except: continue

    fig.update_layout(
        title_text=f'ORR',
        yaxis_title='free energy at U=1.23[eV]',
    )
    
    folder_exist('plots')
    fig.write_html(f'plots/plot_reaction_coord_orr.html', auto_open=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=pathlib.Path, nargs='+')
    args = parser.parse_args()
    
    main(args.db_dir)
