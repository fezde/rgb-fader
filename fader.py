import tempfile
from glob import glob
from logging import getLogger
from math import floor
from os import path

import numpy as np
from moviepy.editor import ImageClip, ImageSequenceClip, concatenate_videoclips
from PIL import Image
from tqdm import tqdm

logger = getLogger(__name__)


class Fader():
    def __init__(self, file_a: str, file_b: str, duration_fade: float = 1, duration_static_image: int = 1) -> None:
        self.fps = 25

        file_a = file_a.strip()
        file_b = file_b.strip()
        if not path.isfile(file_a):
            raise ValueError(f"file_a does not point to a file: {file_a}")
        if not path.isfile(file_b):
            raise ValueError(f"file_b does not point to a file: {file_b}")

        self.data_a = np.asarray(Image.open(file_a))
        self.data_b = np.asarray(Image.open(file_b))
        self.data_current = np.copy(self.data_a)
        # self.data_current.setflags(write=1)

        if self.data_a.shape != self.data_b.shape:
            raise ValueError("Dimensions of images do not match")

        if duration_fade < 1:
            raise ValueError("Fading has to be at least 1 second")

        steps_fade = floor(duration_fade * self.fps)
        self.steps_per_channel = [0, 0, 0]
        self.steps_per_channel[0] = floor(steps_fade/3)
        self.steps_per_channel[1] = floor(steps_fade/3)
        self.steps_per_channel[2] = steps_fade - (self.steps_per_channel[0] + self.steps_per_channel[1])

        self.duration_static_image = duration_static_image
        self.image_counter = 0

    def create_movie(self, output_filename: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.debug(f'created temporary directory: {tmpdirname}')
            self.fade(tmpdirname=tmpdirname, do_persist=True)

            clip_img_a = ImageClip(path.join(tmpdirname, "a.png"), duration=self.duration_static_image)
            clip_img_b = ImageClip(path.join(tmpdirname, "b.png"), duration=self.duration_static_image)

            files = sorted(glob(path.join(tmpdirname, "a2b*.png")))
            clip_fade = ImageSequenceClip(files, fps=self.fps)

            final_clip = concatenate_videoclips([clip_img_a, clip_fade, clip_img_b])
            final_clip.write_videofile(output_filename, fps=self.fps)
            print()

    def fade(self, tmpdirname: str = None, do_persist: bool = False):
        if do_persist:
            Image.fromarray(self.data_a).save(path.join(tmpdirname, "a.png"))
            Image.fromarray(self.data_b).save(path.join(tmpdirname, "b.png"))
            print()
        self.__fade_channel(0, tmpdirname, do_persist=do_persist)
        self.__fade_channel(1, tmpdirname, do_persist=do_persist)
        self.__fade_channel(2, tmpdirname, do_persist=do_persist)

    def __fade_channel(self, channel_idx: int, tmpdirname: str = None, do_persist: bool = False):
        step_size = 1/(self.steps_per_channel[channel_idx]-1)
        for i in tqdm(range(self.steps_per_channel[channel_idx]), desc=f"Rendering channel {channel_idx}"):
            self.__mix_images(channel_idx, step_size*i, tmpdirname, do_persist=do_persist)

    def __mix_images(self, channel_idx: int, ratio_b: float, tmpdirname: str = None, do_persist: bool = False):
        slice_a = self.data_a[:, :, channel_idx]
        slice_b = self.data_b[:, :, channel_idx]
        mix = slice_a*(1-ratio_b) + slice_b*ratio_b
        self.data_current[:, :, channel_idx] = mix

        if do_persist:
            Image.fromarray(self.data_current).save(path.join(tmpdirname, f"a2b_{str(self.image_counter).zfill(4)}.png"))

        self.image_counter += 1
