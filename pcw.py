from typing import List, Tuple
from matplotlib import colors as mcolors
import yaml
import numpy as np
import pptk
import sys

_default_colors = ('y', 'magenta', 'g', 'r', 'b')


def visualize_pc(pcs: Tuple[np.ndarray], colors: Tuple[str] = (), size: float = 0.0005):
    """
    Visualize pointclouds with pptk viewer
    :param pcs: Collection of pointcloutds in the form [N x 3]
    :param colors: String list of colors for each pointcloud see
    https://matplotlib.org/stable/gallery/color/named_colors.html
    :param size: size of the points
    :return: None
    """
    if not colors:
        colors = _default_colors

    np_colors = [mcolors.to_rgba(c) for c in colors]

    pc = np.vstack(pcs)

    pc_color = np.vstack([np.full(pc.shape, col[:-1]) * 255 for pc, col in zip(pcs, np_colors)])

    view = pptk.viewer(pc, pc_color)

    view.set(point_size=size)


def get_yaml_args(config):
    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)
        return cfg


def read_pc(files):
    return [np.load(pc) for pc in files]


def normalize_pc(pc: np.ndarray, means: np.ndarray, max_distance: float):
    return (pc - means) / max_distance


def render_pc(files: List[str],
              colors: List[str],
              sizes: float,
              squeeze: bool,
              normalize: bool):
    """
    Read pcs from files, apply argument options and
    :return:
    """

    pc_list: List[np.ndarray] = []

    if files:
        pc_list += [read_pc([i]) for i in files]

    pc_list = [item for sublist in pc_list for item in sublist]

    if squeeze:
        pc_list = [pc.squeeze(0) for pc in pc_list]

    if normalize:
        reference = pc_list[0]
        means = reference.mean(axis=0, keepdims=True)
        distance = np.sqrt(np.power(reference, 2).sum(axis=1))
        max_distance = np.max(distance)
        pc_list = [normalize_pc(pc, means, max_distance) for pc in pc_list]

    visualize_pc(pc_list, colors, sizes)


if __name__ == '__main__':
    config_file = sys.argv[1]
    cfg = get_yaml_args(config_file)
    render_pc(**cfg)
