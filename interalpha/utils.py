import itertools

import numpy as np


def tokenize(data):
    direct_tokens = (";", "(", ")")
    it = iter(data)
    while True:
        c = next(it)
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


def load_moves(data):
    b_board = np.zeros((19, 19))
    w_board = np.zeros((19, 19))
    yield b_board, w_board, "B"
    for t in tokenize(data):
        if isinstance(t, str):
            continue
        name, value = t
        print(name)
        if name == "B":
            b_board[parse_pos(value)] = 1.0
            yield b_board.copy(), w_board.copy(), "W"
        if name == "W":
            w_board[parse_pos(value)] = 1.0
            yield b_board.copy(), w_board.copy(), "B"


def take(n, iterable):
    return list(itertools.islice(iterable, n))


def load_sgf(filename):
    with open(filename) as f:
        data = f.read()
    yield from load_moves(data)


def print_board(b_board, w_board):
    hoshi = (3, 9, 15)
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
        print("".join(line))
