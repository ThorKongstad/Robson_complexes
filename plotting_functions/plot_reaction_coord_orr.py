import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback
from re import match, search

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from Robson_complexes import sanitize, folder_exist, build_pd
#from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def reaction_energy_traces(
    state_energies: dict,
    plateau_width: float = 0.6,
    connector_dx: float = 0.25,
    plateau_line_width: int = 4,
    connector_line_width: int = 2,
    trace_kwargs: dict = {},
):
    """
    Create Plotly traces for a reaction energy diagram.

    Parameters
    ----------
    state_energies : dict
        Dictionary of {state_name: energy}. Order is preserved.
    plateau_width : float
        Horizontal width of each energy plateau.
    connector_dx : float
        Horizontal slant for connectors (non-vertical transitions).
    plateau_line_width : int
        Line width for plateaus.
    connector_line_width : int
        Line width for connectors.

    Returns
    -------
    tuple
        (plateau_trace, connector_trace)
    """

    states = list(state_energies.keys())
    energies = list(state_energies.values())

    # -----------------------------
    # Plateaus
    # -----------------------------
    x_plateau, y_plateau = [], []

    for i, energy in enumerate(energies):
        left = i - plateau_width / 2
        right = i + plateau_width / 2

        x_plateau.extend([left, right, None])
        y_plateau.extend([energy, energy, None])

    state_kwargs = dict(line=dict(width=plateau_line_width),
        hoverinfo="none",
        name="States",) | trace_kwargs

    plateau_trace = go.Scatter(
        x=x_plateau,
        y=y_plateau,
        mode="lines",
        **state_kwargs
    )

    # -----------------------------
    # Connectors
    # -----------------------------
    x_conn, y_conn = [], []

    for i in range(len(energies) - 1):
        x_start = i + plateau_width / 2 + connector_dx / 2
        x_end = (i + 1) - plateau_width / 2 - connector_dx / 2

        x_conn.extend([x_start, x_end, None])
        y_conn.extend([energies[i], energies[i + 1], None])

    connector_kwargs = dict(line=dict(width=connector_line_width, color="grey"),
        hoverinfo="none",
        name="Transitions",
        showlegend=False ) | trace_kwargs

    connector_trace = go.Scatter(
        x=x_conn,
        y=y_conn,
        mode="lines",
        **connector_kwargs
    )

    return plateau_trace, connector_trace


def main(db_dir: Sequence[str | pathlib.Path], verbose: bool, save_key: Optional[str] = None):
    pd_catalysts = build_pd(db_dir)

    unique_catalysts = {struc for _, row in pd_catalysts.iterrows() if not pd.isna((struc := row.get('structure_str')))}
    unique_charges = {charg for _, row in pd_catalysts.iterrows() if not pd.isna((charg := row.get('gpaw_charge')))}
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

    base_ts = 0
    base_ZPE = 0
    base_ooh_ts = 0.00
    base_ooh_ZPE = 0.39
    base_oh_ts = 0.00
    base_oh_ZPE = 0.30
    base_o_ts = 0.00
    base_o_ZPE = 0.07

    U = 1.23
    step_size = 0.01

    fig = go.Figure()
    for U in np.arange(0, 5, step_size):
        for xc in unique_functionals:
            for cat in unique_catalysts:
                for charge in unique_charges:
                    try:
                        dft_E_None = pd_catalysts.query('structure_str == @cat and gpaw_charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "None"').get('energy').iloc[0]
                        dft_E_O = pd_catalysts.query('structure_str == @cat and gpaw_charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "O"').get('energy').iloc[0]
                        dft_E_OH = pd_catalysts.query('structure_str == @cat and gpaw_charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "OH"').get('energy').iloc[0]
                        dft_E_OOH = pd_catalysts.query('structure_str == @cat and gpaw_charge == @charge and xc == @xc and energy.notna() and adsorbate_str == "OOH"').get('energy').iloc[0]

                        G_None = dft_E_None + base_ts + base_ZPE
                        G_O = dft_E_O + base_o_ts + base_o_ZPE
                        G_OH = dft_E_OH + base_oh_ts + base_oh_ZPE
                        G_OOH = dft_E_OOH + base_ooh_ts + base_ooh_ZPE

                        state_1 = G_None + G_o2 + 4*G_h2/2
                        state_2 = G_OOH + 3*G_h2/2 - 0.3
                        state_3 = G_O + 2*G_h2/2 + G_h2o
                        state_3_alt = G_O + 2*G_h2/2
                        state_4 = G_OH + G_h2/2 + G_h2o - 0.3
                        state_5 = G_None + 2*G_h2o

                        #fig.add_trace(go.Scatter(
                        #    mode='lines',
                        #    name=f'{xc}-{cat}-charge:{charge}-U:{U:.2f}',
                        #    x=('O2', '*OOH', '*O', '*OH', '*'),
                        #    y=(4.92-4*U, state_2-state_1+4.92-3*U, state_3-state_1+4.92-2*U, state_4-state_1+4.92-U, 0),
                        #    legendgrouptitle_text=cat,
                        #    legendgroup=cat,
                        #    visible=False,
                        #))

                        states = {'O2':4.92-4*U, '*OOH':state_2-state_1+4.92-3*U, '*O':state_3-state_1+4.92-2*U, '*OH':state_4-state_1+4.92-U, '*':0}

                        fig.add_traces(
                            reaction_energy_traces(states,
                                                   trace_kwargs=dict(name=f'{xc}-{cat}-charge:{charge}-U:{U:.2f}',
                                                                     legendgrouptitle_text=cat,
                                                                     legendgroup=cat,
                                                                     visible=False,
                                                                     )
                                                   ),
                        )

                    except:
                        if verbose:
                            print(traceback.format_exc())

    fig.update_layout(
        title_text=f'ORR',
        yaxis_title='free energy',
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(states))),
        ticktext=list(states.keys()),
        title="Reaction Coordinate"
    )

    for trace in fig.data:
        if search(r'-U:1\.23', trace.name): trace.visible = True

    steps = []
    for i in np.arange(0, 5, step_size):
        step = dict(
            method='update',
            args=[{'visible': [True if search(f'-U:{i:.2f}', trace.name) else False for trace in fig.data]},
                  {'title_text': f'ORR; U = {i}'}],
            label=f'{i:.2f}'
        )
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "U[eV]= "},
        pad={'t': 50},
        steps=steps,
    )]

    fig.update_layout(sliders=sliders)

    folder_exist('plots')
    fig.write_html(f'plots/plot_reaction_coord_orr_{save_key}.html', auto_open=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=pathlib.Path, nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--save_key', type=str, default=None)
    args = parser.parse_args()
    
    main(args.db_dir, args.verbose, args.save_key)
