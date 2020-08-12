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


def create_leela_input_from_boards(
    boards_black, boards_white, black_move: bool
) -> np.ndarray:
    boards_black = list(reversed(boards_black))
    boards_white = list(reversed(boards_white))
    while len(boards_black) < 8:
        boards_black.append(boards_black[-1])
        boards_white.append(boards_white[-1])

    if black_move:
        lst = boards_black + boards_white + [np.ones((19, 19)), np.zeros((19, 19))]
    else:
        lst = boards_white + boards_black + [np.zeros((19, 19)), np.ones((19, 19))]
    return np.moveaxis(np.array(lst), 0, -1)


def create_leela_input(sgf_nodes, move: int) -> np.ndarray:
    print(move)
    print(np.sum(sgf_nodes[move][0]), len(sgf_nodes))
    nodes = sgf_nodes[max(0, move - 7) : move + 1]
    print(np.sum(nodes[0][0]), len(nodes))
    player = nodes[0][2]
    n0 = [n[0] for n in nodes]
    n1 = [n[1] for n in nodes]

    return create_leela_input_from_boards(n0, n1, player == "B")
