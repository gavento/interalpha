import logging

import click

import tensorflow.keras as K
from interalpha import plot, sgf_utils

from . import cli

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug", "-d", is_flag=True, help="Increase logging verbosity.")
def main(debug):
    logging.getLogger("interalpha").setLevel(
        level=logging.DEBUG if debug else logging.INFO
    )


@cli.main.command()
@click.argument("model", type=click.Path(exists=True))
@click.argument("game", type=click.Path(exists=True))
@click.argument("move", type=int)
def saliency(model, game, move):

    log.debug(f"Loading weights from {game}")
    sgf = list(sgf_utils.load_sgf(game))
    log.debug(f"Create boards at move {move}")
    in0 = sgf_utils.create_leela_input(sgf, move)

    log.debug(f"Loading model {model}")
    m = K.models.load_model(model)

    layers = dict((layer.name, layer) for layer in m.layers)
    name = "residual_1_15_conv_block"
    plot.show_layer(m, layers[name], in0)

    return

    policy, value = model.predict(inp)
    policy = policy[0]
    value = value[0]

    print("PASS:", policy[-1])
    actions = policy[:-1].reshape((19, 19))
    plt.imshow(np.moveaxis(actions, 0, -1))
    plt.show()
