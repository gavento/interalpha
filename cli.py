import logging
import os

import click
import colorlog


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(thin_white)s%(asctime)s%(reset)s [%(log_color)s%(levelname).1s%(reset)s] %(name)s %(message)s"
        )
    )
    colorlog.getLogger().addHandler(handler)
    from interalpha.scripts import cli

    cli.main()


if __name__ == "__main__":
    main()
