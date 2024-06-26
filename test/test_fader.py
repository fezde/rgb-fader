import unittest
from unittest.mock import ANY, call, patch

import numpy as np

from fader import Fader


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestFader(unittest.TestCase):

    def test_file_a_not_found(self):
        with self.assertRaises(ValueError):
            f1 = Fader("a.jpg", "test/b.jpg", 10)

        f2 = Fader("test/a.jpg", "test/b.jpg", 10)
        self.assertIsNotNone(f2)

    def test_file_b_not_found(self):
        with self.assertRaises(ValueError):
            f1 = Fader("test/a.jpg", "b.jpg", 10)

        f2 = Fader("test/a.jpg", "test/b.jpg", 10)
        self.assertIsNotNone(f2)

    def test_enough_steps(self):
        with self.assertRaises(ValueError):
            f1 = Fader("test/a.jpg", "test/b.jpg", 0.5)

        f2 = Fader("test/a.jpg", "test/b.jpg", 2)
        self.assertIsNotNone(f2)

    def test_steps_calculation(self):
        f = Fader("test/a.jpg", "test/b.jpg", 1)
        self.assertEqual(f.steps_per_channel[0], 8)
        self.assertEqual(f.steps_per_channel[1], 8)
        self.assertEqual(f.steps_per_channel[2], 9)

        f = Fader("test/a.jpg", "test/b.jpg", 1.1)
        self.assertEqual(f.steps_per_channel[0], 9)
        self.assertEqual(f.steps_per_channel[1], 9)
        self.assertEqual(f.steps_per_channel[2], 9)

        f = Fader("test/a.jpg", "test/b.jpg", 1.2)
        self.assertEqual(f.steps_per_channel[0], 10)
        self.assertEqual(f.steps_per_channel[1], 10)
        self.assertEqual(f.steps_per_channel[2], 10)

    @patch('fader.Fader._Fader__mix_images')
    @patch('fader.tqdm', notqdm)
    def test_fade_channel_minimal(self, mock_object):
        f = Fader("test/a.jpg", "test/b.jpg", 1, 1)
        f._Fader__fade_channel(0, "", False)

        self.assertEqual(mock_object.call_count, 8)
        mock_object.assert_any_call(0, 0, ANY, do_persist=ANY)
        mock_object.assert_any_call(0, 1, ANY, do_persist=ANY)

    @patch('fader.Fader._Fader__mix_images')
    @patch('fader.tqdm', notqdm)
    def test_fade_channel_realistic_channel_0(self, mock_object):
        f = Fader("test/a.jpg", "test/b.jpg", 107)
        f._Fader__fade_channel(0)

        self.assertEqual(mock_object.call_count, 891)
        mock_object.assert_any_call(0, 0, ANY, do_persist=ANY)
        mock_object.assert_any_call(0, 1, ANY, do_persist=ANY)

    @patch('fader.Fader._Fader__mix_images')
    @patch('fader.tqdm', notqdm)
    def test_fade_channel_realistic_channel_1(self, mock_object):
        f = Fader("test/a.jpg", "test/b.jpg", 107)
        f._Fader__fade_channel(1)

        self.assertEqual(mock_object.call_count, 891)
        mock_object.assert_any_call(1, 0, ANY, do_persist=ANY)
        mock_object.assert_any_call(1, 1, ANY, do_persist=ANY)

    @patch('fader.Fader._Fader__mix_images')
    @patch('fader.tqdm', notqdm)
    def test_fade_channel_realistic_channel_2(self, mock_object):
        f = Fader("test/a.jpg", "test/b.jpg", 107)
        f._Fader__fade_channel(2)

        self.assertEqual(mock_object.call_count, 893)
        mock_object.assert_any_call(2, 0, ANY, do_persist=ANY)
        mock_object.assert_any_call(2, 1, ANY, do_persist=ANY)

    def test_mix_1_channel_1_step_05(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.asarray([
            [
                [127, 0, 0], [127, 0, 0],
            ],
            [
                [127, 0, 0],  [127, 0, 0],
            ],
            [
                [127, 0, 0], [127, 0, 0],
            ],
        ])

        fader._Fader__mix_images(0, 0.5, '', do_persist=False)
        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    def test_mix_1_channel_2_steps(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.asarray([
            [
                [127, 0, 0], [127, 0, 0],
            ],
            [
                [127, 0, 0],  [127, 0, 0],
            ],
            [
                [127, 0, 0], [127, 0, 0],
            ],
        ])

        fader._Fader__mix_images(0, 0.25)
        fader._Fader__mix_images(0, 0.5)
        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    def test_mix_2_channels_1_step(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.asarray([
            [
                [127, 96, 0], [127, 96, 0],
            ],
            [
                [127, 96, 0],  [127, 96, 0],
            ],
            [
                [127, 96, 0], [127, 96, 0],
            ],
        ])

        fader._Fader__mix_images(0, 0.5)
        fader._Fader__mix_images(1, 0.75)
        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    @patch('fader.tqdm', notqdm)
    def test_fade_1_channel(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.asarray([
            [
                [0, 0, 42], [0, 0, 42],
            ],
            [
                [0, 0, 42],  [0, 0, 42],
            ],
            [
                [0, 0, 42], [0, 0, 42],
            ],
        ])

        fader._Fader__fade_channel(2)

        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    @patch('fader.tqdm', notqdm)
    def test_fade_2_channels(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.asarray([
            [
                [0, 128, 42], [0, 128, 42],
            ],
            [
                [0, 128, 42],  [0, 128, 42],
            ],
            [
                [0, 128, 42], [0, 128, 42],
            ],
        ])

        fader._Fader__fade_channel(2)
        fader._Fader__fade_channel(1)

        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    @patch('fader.tqdm', notqdm)
    def test_fade_3_channels(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 10)
        fader.data_a = np.asarray([
            [
                [0, 0, 0], [0, 0, 0],
            ],
            [
                [0, 0, 0],  [0, 0, 0],
            ],
            [
                [0, 0, 0], [0, 0, 0],
            ],
        ])
        fader.data_b = np.asarray([
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42], [255, 128, 42],
            ],
            [
                [255, 128, 42],  [255, 128, 42],
            ],
        ])
        fader.data_current = np.copy(fader.data_a)

        expected = np.copy(fader.data_b)

        fader._Fader__fade_channel(0)
        fader._Fader__fade_channel(2)
        fader._Fader__fade_channel(1)

        self.assertEqual(fader.data_current.tolist(), expected.tolist())

    @patch('fader.tqdm', notqdm)
    def test_fade_3_channels_real_images(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 2)

        fader._Fader__fade_channel(0)
        fader._Fader__fade_channel(1)
        fader._Fader__fade_channel(2)

        self.assertEqual(fader.data_current.tolist(), fader.data_b.tolist())

    @patch('fader.tqdm', notqdm)
    def test_fade_default(self):
        fader = Fader("test/a.jpg", "test/b.jpg", 2)
        fader.fade()

        self.assertEqual(fader.data_current.tolist(), fader.data_b.tolist())
