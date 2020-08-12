import logging

import click
import numpy as np

import plotly.graph_objects as go
import tensorflow as tf
import tensorflow.keras as K
from interalpha import plot, sgf_utils
from plotly.subplots import make_subplots
from typing import List

from . import cli

log = logging.getLogger(__name__)

SCALE_BW = [(0.0, "rgb(0,0,0)"), (1.0, "rgb(255,255,255)")]
SCALE_YG = [(0.0, "rgb(180,180,60)"), (1.0, "rgb(40,230,40)")]
SCALE_RYG = [(0.0, "rgb(230,40,40)"), (0.5, "rgb(180,180,60)"), (1.0, "rgb(40,230,40)")]
SCALE_RGrG = [
    (0.0, "rgb(230,40,40)"),
    (0.5, "rgb(127,127,127)"),
    (1.0, "rgb(40,230,40)"),
]


@click.group()
@click.option("--debug", "-d", is_flag=True, help="Increase logging verbosity.")
def main(debug):
    logging.getLogger("interalpha").setLevel(
        level=logging.DEBUG if debug else logging.INFO
    )


def compute_input_gradient(model, inputs, moves_grad, wins_grad):
    moves_grad = tf.cast(moves_grad, tf.float32)
    wins_grad = tf.cast(wins_grad, tf.float32)
    inputs = tf.cast(inputs, tf.float32)
    assert moves_grad.shape[1] == 362
    assert len(moves_grad.shape) == 2
    assert len(wins_grad.shape) == 1
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        moves, wins = model(inputs)

    input_grads = tape.gradient((moves, wins), inputs, (moves_grad, wins_grad))
    return input_grads, moves, wins


def plot_board(input, axisno=1) -> go.Heatmap:
    assert input.shape == (19, 19, 18)
    black_move = bool(input[0, 0, 16] > 0.5)
    return go.Heatmap(
        z=(input[:, :, 0] - input[:, :, 8]) * (1 - 2 * black_move),
        zmin=-1.0,
        zmax=1.0,
        colorscale=SCALE_BW,
        showscale=False,
        xaxis=f"x{axisno}",
        yaxis=f"y{axisno}",
        name="Black to play" if black_move else "White to play",
    )


def plot_stone_satisfaction(input, grad_win_by_input, axisno=1):
    black_move = bool(input[0, 0, 16] > 0.5)
    p0_in = input[:, :, 0]
    p0_grad = grad_win_by_input[:, :, 0]
    p1_in = input[:, :, 8]
    p1_grad = grad_win_by_input[:, :, 8]

    contrib = p0_grad * p0_in - p1_grad * p1_in
    ext = max([np.max(contrib), -np.min(contrib), 0.0])
    return go.Heatmap(
        z=np.array(contrib),
        zmin=-ext,
        zmax=ext,
        colorscale=SCALE_RGrG,
        showscale=False,
        xaxis=f"x{axisno}",
        yaxis=f"y{axisno}",
    )


def plot_best_moves(move, top=1, axisno=1) -> go.Scatter:
    probs = tf.nn.softmax(move)
    mps = [(probs[y * 19 + x], x, y) for x in range(19) for y in range(19)]
    mps = sorted(mps, reverse=True)[:top]
    return go.Scatter(
        x=[m[1] for m in mps],
        y=[m[2] for m in mps],
        mode="markers",
        marker=dict(
            color=[float(m[0] / mps[0][0]) for m in mps], colorscale=SCALE_YG, size=9
        ),
        text=[f"P={m[0]:.3f} (max={mps[0][0]:3f})" for m in mps],
    )


@cli.main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("game", type=click.Path(exists=True))
@click.argument("move", type=int)
def saliency(model_path, game, move):

    log.debug(f"Loading weights from {game}")
    sgf = list(sgf_utils.load_sgf(game))
    log.debug(f"Create boards at move {move}")
    in0 = sgf_utils.create_leela_input(sgf, move)
    log.debug(f"Loading model {model_path}")
    model = K.models.load_model(model_path)

    input_grads, moves, wins = compute_input_gradient(
        model, [in0], tf.zeros((1, 362)), tf.ones(1)
    )
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(plot_board(in0), row=1, col=1)
    fig.add_trace(plot_best_moves(moves[0], top=5), row=1, col=1)
    fig.add_trace(plot_stone_satisfaction(in0, input_grads[0], axisno=2), row=1, col=2)
    fig["layout"]["yaxis"]["scaleanchor"] = "x"
    fig["layout"]["yaxis2"]["scaleanchor"] = "x2"
    fig.show()

    return

    layers = dict((layer.name, layer) for layer in model.layers)
    name = "residual_1_15_conv_block"
    fig = plot.plot_layer(model, layers[name], in0)
    fig.show()

    return

    policy, value = model.predict(inp)
    policy = policy[0]
    value = value[0]

    print("PASS:", policy[-1])
    actions = policy[:-1].reshape((19, 19))
    plt.imshow(np.moveaxis(actions, 0, -1))
    plt.show()
