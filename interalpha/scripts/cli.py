import logging
from typing import List

import click
import numpy as np

import plotly.graph_objects as go
import tensorflow as tf
import tensorflow.keras as K
from interalpha import plot, sgf_utils
from plotly.subplots import make_subplots

from .. import leela_situation
from . import cli

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug", "-d", is_flag=True, help="Increase logging verbosity.")
def main(debug):
    logging.getLogger("interalpha").setLevel(
        level=logging.DEBUG if debug else logging.INFO
    )


def add_go_grid(fig, row, col, axisno=1):
    for i in [3, 4, 9, 10, 15, 16]:
        fig.add_shape(
            dict(type="line", x0=i - 0.5, x1=i - 0.5, y0=-0.5, y1=18.5),
            row=row,
            col=col,
            xref=f"x{axisno}",
            yref=f"y{axisno}",
        )
        fig.add_shape(
            dict(type="line", y0=i - 0.5, y1=i - 0.5, x0=-0.5, x1=18.5),
            row=row,
            col=col,
            xref=f"x{axisno}",
            yref=f"y{axisno}",
        )


def fig_for_move(s: leela_situation.LeelaSituation, move=None, prob=None):
    fig = make_subplots(rows=1, cols=3)
    for i in range(1, 4):
        fig["layout"][f"yaxis{i}"]["scaleanchor"] = f"x{i}"
        add_go_grid(fig, 1, i, i)

    g = s.input_grad_for_move(move)
    l2_per_time = np.mean(np.mean(g ** 2, axis=0), axis=0) ** 0.5
    log.debug(
        f"Active player mean gradients: {' '.join(f'{x:.2g}' for x in l2_per_time[0:8])}"
    )
    log.debug(
        f"Second player mean gradients: {' '.join(f'{x:.2g}' for x in l2_per_time[8:16])}"
    )
    log.debug(f"Black / white active mean gradients: {l2_per_time[16:18]}")

    # Board
    fig.add_trace(s.plotly_board(), row=1, col=1)
    if move is None:
        fig.add_trace(s.plotly_all_moves(), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[move[0]], y=[move[1]], mode="markers"))

    # Saliency
    fig.add_trace(
        s.plotly_saliency(
            move=move, axisno=2, opacity=1, scale=leela_situation.SCALE_BBe
        ),
        row=1,
        col=2,
    )

    # Reinforce color (normalized)
    fig.add_trace(s.plotly_color_preference(move=move, axisno=3), row=1, col=3)

    if move is None:
        title = (
            f"{str(s)}: move probs. | win prob. saliency map | win reinforcing gradient"
        )
    else:
        title = f"Move {move} with P={prob:.3f}: situation | saliency map of move | move reinforcing gradient"

    fig.update_layout(
        title=title,
        autosize=False,
        width=1100,
        height=320,
        margin=dict(l=10, r=10, b=10, t=30, pad=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figures_to_html(figs, filename, include_plotlyjs="cdn"):
    with open(filename, "w") as f:
        f.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = (
                fig.to_html(include_plotlyjs=include_plotlyjs)
                .split("<body>")[1]
                .split("</body>")[0]
            )
            f.write(inner_html)
        f.write("</body></html>" + "\n")


@cli.main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("game_path", type=click.Path(exists=True))
@click.argument("turn", type=int)
@click.option("-t", "--top-moves", default=5)
@click.option("-o", "--output", default="plot-saliency.html")
def saliency(model_path, game_path, turn, top_moves, output):
    model = K.models.load_model(model_path)
    s = leela_situation.LeelaSituation.load_sgf(model, game_path, turn)
    log.info(f"Summary: {s.summary()}")

    figs = [
        fig_for_move(s, move=move[0], prob=move[1])
        for move in [(None, None)] + s.top_moves[:top_moves]
    ]
    figures_to_html(figs, output)

