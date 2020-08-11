import itertools
import logging

import numpy as np

log = logging.getLogger(__name__)


def tokenize(data):
    direct_tokens = (";", "(", ")")
    it = iter(data)
    for c in it:
        if c.isalpha():
            buffer = [c]
            c = next(it)
            while c != "[":
                buffer.append(c)
                c = next(it)
            name = "".join(buffer)
            buffer = []
            c = next(it)
            while c != "]":
                buffer.append(c)
                c = next(it)
            value = "".join(buffer)
            yield name, value
        elif c in direct_tokens:
            yield c
        elif c.isspace():
            continue
        else:
            raise Exception("Invalid symbol: ", c)


def parse_pos(poscode):
    poscode = poscode.lower()
    return ord(poscode[0]) - ord("a"), ord(poscode[1]) - ord("a")


def load_moves(data) -> list:
    b_board = np.zeros((19, 19))
    w_board = np.zeros((19, 19))
    res = [(b_board, w_board, "B")]
    for t in tokenize(data):
        if isinstance(t, str):
            continue
        name, value = t
        if name == "B":
            b_board[parse_pos(value)] = 1.0
            res.append((b_board.copy(), w_board.copy(), "W"))
        if name == "W":
            w_board[parse_pos(value)] = 1.0
            res.append((b_board.copy(), w_board.copy(), "B"))
    return res


def take(n, iterable):
    return list(itertools.islice(iterable, n))


def load_sgf(filename):
    with open(filename) as f:
        data = f.read()
    return load_moves(data)


def print_board(b_board: np.ndarray, w_board: np.ndarray) -> str:
    hoshi = (3, 9, 15)
    res = []
    for j in range(19):
        line = []
        for i in range(19):
            pos = (i, j)
            if b_board[pos] > 0:
                line.append("B")
            elif w_board[pos] > 0:
                line.append("W")
            elif i in hoshi and j in hoshi:
                line.append("+")
            else:
                line.append(".")
        res.append("".join(line))
    return "\n".join(res)


def create_leela_input(sgf_nodes, move: int) -> np.ndarray:
    nodes = sgf_nodes[move : max(0, move - 8) : -1]
    player = nodes[0][2]

    n0 = [n[0] for n in nodes]
    n1 = [n[1] for n in nodes]

    while len(n0) < 8:
        n0.append(n0[-1])
        n1.append(n1[-1])

    log.debug(
        f"=== Player {player}:\n{print_board(nodes[0][0], nodes[0][1])}"
    )
    if player == "B":
        lst = n0 + n1 + [np.ones((19, 19)), np.zeros((19, 19))]
    elif player == "W":
        lst = n1 + n0 + [np.zeros((19, 19)), np.ones((19, 19))]
    else:
        assert 0

    inp = np.array(lst)
    return np.moveaxis(inp, 0, -1)
