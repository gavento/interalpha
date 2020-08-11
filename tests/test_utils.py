from interalpha import utils


def test_load_sgf(datadir):
    sgf = utils.load_sgf(datadir / "37vm-gokifu-20200804-Ke_Jie-Zhao_Chenyu.sgf")
    for i in range(17):
        next(sgf)
    b, w, p = next(sgf)
    utils.print_board(b, w)
