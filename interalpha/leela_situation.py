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


def idx2move(idx):
    assert idx >= 0 and idx < 362
    if idx == 361:
        return "pass"
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
            enumerate(self.move_probs), reverse=True, key=lambda p: p[1]
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

    def __str__(self) -> str:
        return f"Turn {self.turn}, {self.active_player} to move, est. win chance {100*float(self.win_prob):.2f}%"

    def board_str(self):
        return sgf_utils.print_board(self.boards_black[-1], self.boards_white[-1])

    def summary(self) -> str:
        topm = ", ".join(
            f"{idx2move(i)}: {100 * float(p):.2f}%" for i, p in self.top_moves[:5]
        )
        return f"{str(self)}\nTop moves: {topm}\n{self.board_str()}"

