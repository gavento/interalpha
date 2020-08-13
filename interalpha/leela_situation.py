import logging
from typing import List

import click
import numpy as np

import plotly.graph_objects as go
import tensorflow as tf
import tensorflow.keras as K
from interalpha import plot, sgf_utils
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


SCALE_BW = [(0.0, "rgb(0,0,0)"), (1.0, "rgb(255,255,255)")]
SCALE_YG = [(0.0, "rgb(180,180,60)"), (1.0, "rgb(40,230,40)")]
SCALE_RYG = [(0.0, "rgb(230,40,40)"), (0.5, "rgb(180,180,60)"), (1.0, "rgb(40,230,40)")]
SCALE_RGrG = [
    (0.0, "rgb(230,40,40)"),
    (0.5, "rgb(127,127,127)"),
    (1.0, "rgb(40,230,40)"),
]
SCALE_TrYG = [
    (0.0, "rgba(127,127,127,0.0)"),
    (0.1, "rgba(250,250,0,0.6)"),
    (1.0, "rgba(40,250,40,1.0)"),
]
SCALE_TrBe = [
    (0.0, "rgba(0,0,255,0)"),
    (1.0, "rgba(0,0,255,1)"),
]
SCALE_BBe = [
    (0.0, "rgba(0,0,0,1)"),
    (1.0, "rgba(0,0,255,1)"),
]

PASS_MOVE = (19, 0)

SCALE_BGW = [
    (0.0, "rgb(0,0,0)"),
    (0.45, "rgb(115,115,115)"),
    (0.5, "rgb(112,160,112)"),
    (0.55, "rgb(140,140,140)"),
    (1.0, "rgb(255,255,255)"),
]


def idx2movename(idx):
    m = idx2move(idx)
    if m == PASS_MOVE:
        return "pass"
    return m


def idx2move(idx):
    assert idx >= 0 and idx < 362
    if idx == 361:
        return PASS_MOVE
    return (idx % 19, idx // 19)


def move2idx(move):
    if isinstance(move, tuple):
        return 19 * move[1] + move[0]
    elif move == "pass":
        return 361
    else:
        assert 0, str(move)


class LeelaSituation:
    """
    Leela+gradients wrapper around single board situation.

    Turns are numbered from 1.
    `boards_black[T]` is state *after* turn T.
    That is, `boards_black[0]` is empty, `boards_black[1]` contains one (black) stone.
    Same for `boarsd_white`.
    """

    def __init__(
        self,
        model,
        boards_black: List[np.ndarray],
        boards_white: List[np.ndarray],
        black_move: bool = None,
    ):
        self.model = model
        self.boards_black = boards_black
        self.boards_white = boards_white
        self.turn = len(boards_black)
        self.black_move = black_move if black_move is not None else self.turn % 2 == 1
        assert len(self.boards_black) == len(self.boards_white)
        assert all(b.shape == (19, 19) for b in self.boards_black)
        assert all(b.shape == (19, 19) for b in self.boards_white)

        log.debug("Running model and creating gradient tape")
        self.net_input = sgf_utils.create_leela_input_from_boards(
            self.boards_black, self.boards_white, self.black_move
        )
        self.tape_input = tf.cast(self.net_input, tf.float32)
        with tf.GradientTape(persistent=True) as self.tape:
            self.tape.watch(self.tape_input)
            self.model_res = self.model(tf.expand_dims(self.tape_input, 0))
            self.tape_move_logits = self.model_res[0][0]
            self.tape_win_tanh = self.model_res[1][0]
            self.move_probs = tf.nn.softmax(self.tape_move_logits)
            self.win_prob = (1.0 + tf.tanh(self.tape_win_tanh)) / 2.0
        self.top_moves = sorted(
            [(idx2move(i), p) for i, p in enumerate(self.move_probs)],
            reverse=True,
            key=lambda p: p[1],
        )

    @property
    def active_player(self):
        return ["white", "black"][self.black_move]

    @classmethod
    def load_sgf(cls, model, sgf_path, turn):
        assert turn > 0
        sgf = list(sgf_utils.load_sgf(sgf_path))[:turn]
        assert (sgf[-1][2] == "B") == (len(sgf) % 2 == 1)
        return cls(model, [s[0] for s in sgf], [s[1] for s in sgf])

    def compute_input_grad(self, moves_grad, win_grad):
        return self.tape.gradient(
            self.model_res,
            self.tape_input,
            (np.array([moves_grad]), np.array([win_grad])),
        )

    def input_grad_for_move(self, move: int = None):
        "`move=None` for win-prob (tanh) gradient"
        m = np.zeros((362,), dtype=np.float32)
        if move is None:
            return self.compute_input_grad(m, tf.ones(1))
        else:
            m[move2idx(move)] = 1.0
            return self.compute_input_grad(m, tf.zeros(1))

    def __str__(self) -> str:
        return f"Turn {self.turn}, {self.active_player} to move, est. win chance {100*float(self.win_prob):.2f}%"

    def board_str(self):
        return sgf_utils.print_board(self.boards_black[-1], self.boards_white[-1])

    def summary(self) -> str:
        topm = ", ".join(
            f"{'pass' if m == PASS_MOVE else m}: {100 * float(p):.2f}%"
            for m, p in self.top_moves[:5]
        )
        return f"{str(self)}\nTop moves: {topm}\n{self.board_str()}"

    def plotly_board(self, axisno=1) -> go.Heatmap:
        return go.Heatmap(
            z=(self.boards_white[-1] - self.boards_black[-1]),
            zmin=-1.0,
            zmax=1.0,
            colorscale=SCALE_BW,
            showscale=False,
            xaxis=f"x{axisno}",
            yaxis=f"y{axisno}",
            name=str(self),
        )

    def plotly_top_moves(self, top=1, axisno=1) -> go.Scatter:
        t = self.top_moves[:top]
        tx = [-1 if m == "pass" else m[0] for m, _ in t]
        ty = [0 if m == "pass" else m[1] for m, _ in t]
        return go.Scatter(
            x=tx,
            y=ty,
            mode="markers",
            marker=dict(
                color=[float(p / t[0][1]) for _, p in t], colorscale=SCALE_YG, size=8
            ),
            text=[f"P={p:.3f} (max={t[0][1]:.3f})" for _, p in t],
        )

    def plotly_all_moves(self, opacity=0.7, axisno=1) -> go.Heatmap:
        mp = np.reshape(self.move_probs[:361], (19, 19))
        return go.Heatmap(
            z=mp,
            zmin=0.0,
            colorscale=SCALE_TrYG,
            showscale=False,
            xaxis=f"x{axisno}",
            yaxis=f"y{axisno}",
            name=str(self),
            opacity=opacity,
        )

    def plotly_saliency(
        self, move=None, opacity=0.7, axisno=1, scale=SCALE_TrBe
    ) -> go.Heatmap:
        ig = self.input_grad_for_move(move=move)
        vals = np.abs(np.sum(ig[:, :, 0:8], axis=-1)) + np.abs(
            np.sum(ig[:, :, 8:16], axis=-1)
        )
        return go.Heatmap(
            z=np.array(vals),
            zmin=0.0,
            colorscale=scale,
            showscale=False,
            xaxis=f"x{axisno}",
            yaxis=f"y{axisno}",
            opacity=opacity,
        )

    def plotly_color_preference(self, move=None, axisno=1) -> go.Heatmap:
        ig = self.input_grad_for_move(move=move)
        vals = np.sum(ig[:, :, 0:8], axis=-1) - np.sum(ig[:, :, 8:16], axis=-1)
        if self.black_move:
            vals = -vals
        vals = vals - np.mean(vals)
        ext = max([np.max(vals), -np.min(vals), 0.0])
        vals = vals / ext
        return go.Heatmap(
            z=np.array(vals),
            zmin=-1,
            zmax=1,
            colorscale=SCALE_BGW,
            showscale=True,
            xaxis=f"x{axisno}",
            yaxis=f"y{axisno}",
        )

