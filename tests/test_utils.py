from interalpha import sgf_utils

SGF_FILE = "37vm-gokifu-20200804-Ke_Jie-Zhao_Chenyu.sgf"


def test_load_sgf(datadir):
    sgf_nodes = sgf_utils.load_sgf(datadir / SGF_FILE)
    b, w, p = sgf_nodes[15]
    sgf_utils.print_board(b, w)


def test_create_input(datadir):
    sgf_nodes = sgf_utils.load_sgf(datadir / SGF_FILE)
    in0 = sgf_utils.create_leela_input(sgf_nodes, 1)
    assert in0.shape == (19, 19, 18)
    in1 = sgf_utils.create_leela_input(sgf_nodes, 6)
    assert in1.shape == (19, 19, 18)
    in2 = sgf_utils.create_leela_input(sgf_nodes, 15)
    assert in2.shape == (19, 19, 18)

