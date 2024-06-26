
import argparse
from logging import DEBUG, INFO, basicConfig, getLogger
from os import path

import numpy as np

from fader import Fader

basicConfig(level=INFO)
logger = getLogger(__name__)


parser = argparse.ArgumentParser(
    prog='RGB Fader',
    description='Create a fading animation between 2 images by fading each RGB channel seperately',
)

parser.add_argument('file_from', metavar="from", type=argparse.FileType('r'))
parser.add_argument('file_to', metavar='to', type=argparse.FileType('r'))


def main():
    args = parser.parse_args()

    file_from = args.file_from.name
    file_to = args.file_to.name

    if file_from == file_to:
        raise ValueError("You cannot fade between the same file.")

    result_name = f"{path.basename(file_from)}_2_{path.basename(file_to)}.mp4"
    fader = Fader(
        file_a=file_from,
        file_b=file_to,
        duration_fade=4,
        duration_static_image=3)

    fader.create_movie(result_name)


if __name__ == "__main__":
    main()
# end main
