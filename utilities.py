import copy
import math
import typing
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# define the dictionary of refractive indices at different wavelengths
refractive_index_dict = {'fused silica': {400: 1.4701, 450: 1.4656, 500: 1.4623, 550: 1.4599, 600: 1.4580,
                                          650: 1.4565, 700: 1.4553},
                         # fused silica, https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
                         'bk7': {300: 1.5528, 350: 1.5392, 400: 1.5308, 450: 1.5253, 500: 1.5214,
                                 550: 1.5185, 600: 1.5163, 650: 1.5145, 700: 1.5131},
                         # N-BK7, https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
                         'crown': {350: 1.5350, 400: 1.5261, 450: 1.5201, 500: 1.5160, 550: 1.5129,
                                   600: 1.5106, 650: 1.5087, 700: 1.5072},
                         # Schott crown glass, https://refractiveindex.info/?shelf=glass&book=SCHOTT-K&page=K7
                         'flint': {350: 1.6742, 400: 1.6522, 450: 1.6388, 500: 1.6299, 550: 1.6237,
                                   600: 1.6190, 650: 1.6154, 700: 1.6126},
                         # Schott flint glass, https://refractiveindex.info/?shelf=glass&book=SCHOTT-F&page=F2
                         'sapphire': {300: 1.8144, 350: 1.7972, 400: 1.7862, 450: 1.7794, 500: 1.7742,
                                      550: 1.7704, 600: 1.7675, 650: 1.7651, 700: 1.7632}}


def print_md(string):
    display.display(display.Markdown(string))


def refractive_index(material: str, wavelength: float = None):
    """

    :param material: string, valid key of _refractive_indices dictionary
    :param wavelength: float of wavelength at which to interpolate refractive index, default None to return full dict
    :return: dict or float
    """
    n = refractive_index_dict.get(material, None)
    if n is None:
        raise KeyError(f"'material' argument must be one of: {list(refractive_index_dict.keys())}")
    if wavelength is None:
        return n
    else:
        wavelengths = np.array(list(n.keys()))
        ns = np.array(list(n.values()))
        return np.interp(wavelength, wavelengths, ns)


def print_diag(message, diag=True):
    if diag:
        print(message)


class IntersectionExceedsBounds(Exception):
    def __init__(self, message, Pr=()):
        self.message = message
        self.Pr = Pr


class NegativeIntersectionDistances(Exception):
    def __init__(self, message, Prs=()):
        self.message = message
        self.Prs = Prs


def plot_aberrations(rays, unit='rad', failed=True, new_fig=True, c=None, mfc=None, ls=':', prefix='', yloc='in',
                     flip_axes=False):
    from windows import CurvedWall
    from raytrace import RaySet
    if isinstance(rays, CurvedWall):
        rays = rays.rays

    assert isinstance(rays, RaySet)

    if new_fig:
        plt.figure()

    y_pos_func, y_pos_label = {'in': (rays.ys_in, 'Incoming'), 'out': (rays.ys_out, 'Outgoing')}[yloc]

    x, y = (y_pos_func('success'), rays.aberrations(sets='success', unit=unit))[::{True: -1, False: 1}[flip_axes]]
    h = plt.plot(x, y, 'o', c=c, mfc=mfc, ls=ls, label=prefix + {True: 'Passed Rays', False: ''}[failed])

    if failed:
        if c is None:
            c = h[0].get_color()
        x, y = (y_pos_func('failed'), rays.aberrations(sets='failed', unit=unit))[::{True: -1, False: 1}[flip_axes]]
        plt.plot(x, y, 'o', c=c, mfc='w', ls='', label=prefix+'Failed Rays')

    x, y = (y_pos_label + ' Y-Position (cm)', f'Aberration ({unit})')[::{True: -1, False: 1}[flip_axes]]
    plt.ylabel(y)
    plt.xlabel(x)

    plt.legend()


def ray_aberration_subplots():
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 4), sharey='row')
    plt.subplots_adjust(wspace=0)

    axs[1].yaxis.set_label_position("right")
    axs[1].axvline(0, ls=':', c='k')

    return fig, axs
