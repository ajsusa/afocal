import math
import typing
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Union, Sequence

from utilities import print_diag, print_md, refractive_index, NegativeIntersectionDistances, NoIntersection
import raytrace

_n_o = 1.
_n_i = 1.

_length_unit = 'cm'
_power_unit = '1/m'


def power_mult(P_unit=None):
    if P_unit is None:
        P_unit = _power_unit
    return {'1/cm': 1., '1/m': 100.}[P_unit]


def set_n_o(n):
    if n >= 1.:
        global _n_o
        _n_o = float(n)


def set_n_i(n):
    if n >= 1.:
        global _n_i
        _n_i = float(n)


def check_no_ni(n_o, n_i):
    if n_o is None:
        global _n_o
        n_o = _n_o
    if n_i is None:
        global _n_i
        n_i = _n_i
    return n_o, n_i


class CurvedWall(object):
    def __init__(self, Ri: float, h: float, t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = None, n_i: float = None):
        n_o, n_i = check_no_ni(n_o, n_i)
        self._Ri = float(Ri)
        self._h = float(h)
        self._t_1 = float(t)
        self._h_f = float(h_f)
        self._material_1 = material
        if isinstance(design_wavelength, typing.Iterable):
            self._design_wavelength = tuple(design_wavelength)
        else:
            self._design_wavelength = float(design_wavelength)
        self._n_o = float(n_o)
        self._n_i = float(n_i)
        self._rays = None
        self._optimization_parameters = None
        self._optimization_metric = None

    @property
    def Ri(self):
        return self._Ri

    @property
    def Qi(self):
        return np.array([0, 0])

    @property
    def h(self):
        return self._h

    @property
    def t_1(self):
        return self._t_1

    @property
    def h_f(self):
        return self._h_f

    @property
    def material_1(self):
        return self._material_1

    @property
    def design_wavelength(self):
        return self._design_wavelength

    @property
    def n_1(self):
        design_wavelengths = self.design_wavelength
        if isinstance(design_wavelengths, typing.Iterable):
            return [refractive_index(self.material_1, wave) for wave in design_wavelengths]
        else:
            return refractive_index(self.material_1, design_wavelengths)

    @property
    def n_i(self):
        return self._n_i

    @property
    def n_o(self):
        return self._n_o

    @property
    def x_min(self):
        """To be overwritten in sub classes"""
        return 0

    @property
    def x_max(self):
        """To be overwritten in sub classes"""
        return 0

    @property
    def default_x_start(self):
        return -1.1 * abs(self.x_min)

    @property
    def default_x_stop(self):
        return 1.1 * abs(self.x_max)

    @property
    def rays(self):
        return self._rays

    @property
    def Y_max_i(self):
        return self.h / 2 - self.h_f

    @property
    def Y_max_o(self):
        return self.h / 2

    @property
    def estimated_Y_max_in(self):
        wave = self.design_wavelength
        if isinstance(wave, typing.Iterable):
            wave = list(wave)[0]
        ray = raytrace.Ray([0, self.h / 2. - self.h_f], wave, self.n_i)
        self.ray_trace_window(ray, 'right')
        return ray.Pr[1]

    @property
    def default_Y_range(self):
        y0 = self.estimated_Y_max_in
        return -y0, y0

    @property
    def optimization_metric(self):
        return self._optimization_metric

    @optimization_metric.setter
    def optimization_metric(self, metric):
        raise NotImplementedError(f'Optimization not implemented for window of type "{type(self)}"')

    @property
    def optimization_parameters(self):
        if self._optimization_parameters is None:
            return self.default_optimization_parameters
        return self._optimization_parameters

    @optimization_parameters.setter
    def optimization_parameters(self, parameters):
        if parameters is None:
            parameters = self.optimization_parameters
        if all([p in self.all_optimization_parameters for p in parameters]):
            self._optimization_parameters = parameters
        else:
            bad_params = [p for p in parameters if p not in self.all_optimization_parameters]
            raise ValueError(f'"optimization_parameters" {bad_params} not in valid set'
                             f' {self.all_optimization_parameters}')

    @property
    def default_optimization_parameters(self):
        return None

    @property
    def all_optimization_parameters(self):
        return None

    @property
    def default_bounds(self):
        return {'Rm': (-1.8 / self.Y_max_o, 1.8 / self.Y_max_o),
                'Ro': (-1.8 / self.Y_max_o, 1.8 / self.Y_max_o),
                't_1': {self.h / 20., None},
                't_2': (self.h / 20., None)}

    def inplace(self, inplace):
        return {False: copy.deepcopy(self)}.get(inplace, self)

    def print_dims(self, prefix=''):
        print(prefix + f"Ri =\t{self.Ri:6.3f} {_length_unit}")
        print(prefix + f"t =\t{self.t_1:6.3f} {_length_unit}")

    def draw(self, which='both', n_pts=100, center_circle=True, ls='-', c='k', lw=2, new_figure=True, figsize=(10, 5)):
        pass

    def draw_annulus(self, c='k', ls='-', lw=1.5):
        pass

    def get_n1(self, wavelength=None):
        if wavelength is None:
            return self.n_1
        return refractive_index(self.material_1, wavelength=wavelength)

    def ray_trace_functions(self, side: str = 'left'):
        funcs = self.ray_trace_exterior, self.ray_trace_middle, self.ray_trace_interior
        if side == 'left':
            return funcs
        elif side == 'right':
            return funcs[::-1]

    def ray_trace_interior(self, ray: raytrace.Ray, side: str, diagnostics=False):
        print_diag('  Tracing the inner surface', diagnostics)
        n1 = self.get_n1(ray.wavelength)
        if side not in ['both', 'left', 'right']:
            pass
        elif side == 'left':
            ray.refract_circle(self.Ri, self.Qi, self.n_i, Y_max=self.Y_max_i, diagnostics=diagnostics)
        elif side == 'right':
            ray.refract_circle(self.Ri, self.Qi, n1, Y_max=self.Y_max_i, diagnostics=diagnostics)
        elif side == 'both':
            ray.refract_circle(self.Ri, self.Qi, self.n_i, Y_max=self.Y_max_i, diagnostics=diagnostics)
            ray.refract_circle(self.Ri, self.Qi, n1, Y_max=self.Y_max_i, diagnostics=diagnostics)

    def ray_trace_middle(self, ray: raytrace.Ray, side: str, diagnostics=False):
        # placeholder for doublet windows
        pass

    def ray_trace_exterior(self, ray: raytrace.Ray, side: str, diagnostics=False):
        # placeholder to be replaced with spherical or ashperical surface
        pass

    def ray_trace_window(self, ray: raytrace.Ray, side, diagnostics=False, pass_error=True):
        if side not in ['left', 'right']:
            raise ValueError('"Side" must be one of "left" or "right".')
        else:
            print_diag(f'Tracing the {side} window', diagnostics)
            trace_functions = self.ray_trace_functions(side)
            for f in trace_functions:
                try:
                    f(ray, side, diagnostics)
                except NoIntersection as err:  # if the ray doesn't intersect...
                    if pass_error:  # move on to next surface
                        pass
                    else:  # raise an exception
                        err.ray = ray
                        raise err

    def ray_trace_windows(self, ray: raytrace.Ray, x_stop=None, dx_extend=None, diagnostics=True):
        self.ray_trace_window(ray, 'left', diagnostics=diagnostics)
        self.ray_trace_window(ray, 'right', diagnostics=diagnostics)

        if x_stop is not None:
            ray.extend_to_x(x_stop)
        elif dx_extend is not None:
            ray.extend_dx(dx_extend)

    def trace_ray_set(self, wavelength: Union[float, int, Sequence] = None, n_rays=21, y_range=None, x_start=None, x_stop=None, dx_extend=None,
                      append=False, color='r', diagnostics=False, draw=True, arrow_size=10, inplace=True, marker='',
                      linestyle=None):
        if append and inplace:
            ray_set = self._rays
        else:
            ray_set = raytrace.RaySet()

        if wavelength is None:
            wavelength = self.design_wavelength

        print_diag(f'y_range =\t{y_range}', False)
        if y_range is None:
            # y_range = -self.h / 2 + self.h_f, self.h / 2 - self.h_f
            y_range = self.default_Y_range
        elif isinstance(y_range, typing.Iterable):
            pass
        else:
            y_range = -np.abs(y_range), np.abs(y_range)
        print_diag(f'y_range =\t{y_range}', False)

        ys_in = np.linspace(y_range[0], y_range[1], n_rays)

        if x_start is None:
            x_start = self.default_x_start
            # x_start = self.x_min - 0.1 * np.abs(self.x_min)

        if x_stop is None and dx_extend is None:
            x_stop = self.default_x_stop
            # x_stop = self.x_max + 0.1 * np.abs(self.x_max)

        if not isinstance(wavelength, typing.Iterable):
            wavelength = [wavelength]
        for wave in wavelength:
            for y in ys_in:
                ray = raytrace.Ray((x_start, y), wave, self.n_o, c=color)
                try:
                    self.ray_trace_windows(ray, x_stop=x_stop, dx_extend=dx_extend, diagnostics=diagnostics)
                except NegativeIntersectionDistances as err:
                    if ray.success:
                        self.draw()
                        if self.rays is not None:
                            [r.draw(c='g', ls='-', marker='.', arrow_size=0) for r in self.rays.rays_of_set('all')]
                        ray.draw(c='r', ls=':', marker='.', arrow_size=0)
                        for Pr in err.Prs:
                            plt.plot(Pr[0], Pr[1], 'or')
                        raise err
                except NoIntersection as err:
                    self.draw()
                    err.draw_ray()
                    raise err
                except ValueError as err:
                    pass
                    # self.draw()
                    # ray.draw()
                    # raise err

                if draw:
                    ray.draw(ls=linestyle, marker=marker, arrow_size=arrow_size)
                ray_set.add_ray(ray)

        if inplace:
            self._rays = ray_set

        return ray_set

    def draw_rays(self, sets='all', c='r', ls='-', marker='', arrow_size=8):
        for ray in self.rays.rays_of_set(sets):
            ray.draw(c=c, ls=ls, marker=marker, arrow_size=arrow_size)

    def check_fix_geometry(self):
        pass

    def set_parameter(self, p, name, preserve_power=True):
        if name == 't1' or name == 't_1':
            self._t_1 = p
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')

    def set_parameters(self, parameter_names, parameter_values, preserve_power=True):
        for name, val in zip(parameter_names, parameter_values):
            self.set_parameter(name, val, preserve_power=preserve_power)

    def get_parameter(self, name):
        if name == 't1' or name == 't_1':
            return self.t_1
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')

    def get_parameters(self, parameter_names, as_array=True):
        ps = [self.get_parameter(p) for p in parameter_names]
        if as_array:
            ps = np.array(ps)
        return ps

    def err(self, ps, diagnostics=False, n_rays=21, preserve_power=True):
        parameters = self.optimization_parameters

        self.set_parameters(parameters, ps, preserve_power=preserve_power)
        # print_diag(f'Iteration {i[0]}', diagnostics)

        rays = self.trace_ray_set(y_range=None, wavelength=self.design_wavelength, n_rays=n_rays,
                                  draw=False, inplace=False)
        error = {'rms_aberration': rays.rms_aberration(sets='all'),
                 'max_aberration': rays.max_aberration(sets='all')}.get(self.optimization_metric, None)
        if error is None:
            raise ValueError(f'Invalid metric "{self.optimization_metric}" for "{type(self)}" optimization.' +
                             'Argument must be one of "rms_aberration" or "max_aberration"')

        print_diag([name + '=' + f'{p}' for name, p in zip(parameters, ps)] + [f'\t error = {error}'], diagnostics)

        return error

    def optimize_window(self, metric='rms_aberration', parameters=None, wavelength=None,
                        bounds_dict=None, inplace=True, n_rays=21, y_range=None, ray_sets='success',
                        preserve_power=True, status=False):

        window = self.inplace(inplace)

        if metric is None:
            metric = window.optimization_metric
        else:
            window.optimization_metric = metric

        if parameters is None:
            parameters = window.default_optimization_parameters
        else:
            window.optimization_parameters = parameters

        err = lambda ps: window.err(ps, diagnostics=status, n_rays=n_rays, preserve_power=preserve_power)

        # default_bounds = {'Rm': (-1.8 / window.Y_max_o, 1.8 / window.Y_max_o),
        #                   'Ro': (-1.8 / window.Y_max_o, 1.8 / window.Y_max_o)}

        if bounds_dict is None:
            bounds_dict = {}
        if type(bounds_dict) is dict:
            bounds = [bounds_dict.get(p, window.default_bounds.get(p, (None, None))) for p in parameters]
        else:
            raise ValueError(f'"bounds_dict" must be dictionary with keys in "parameters" or None')

        if y_range is None:
            y_range = self.default_Y_range

        p0 = window.get_parameters(parameter_names=parameters, as_array=True)

        print_diag(f'p0 =    \t{p0}\nbounds =\t{bounds}', status)
        print_diag(f'error0 =\t{err(p0)}', status)

        sln = minimize(err, p0, jac='2-point', bounds=bounds)

        print_diag(f'Current error value:               {err(sln.x):3.2e}', status)
        # print_diag(f'Minimum error during optimization: {min_error[0]:3.2e}', False)

        window.set_parameters(parameters, sln.x)
        window.trace_ray_set(wavelength=wavelength, n_rays=n_rays, y_range=y_range, draw=False, inplace=True)

        return window, sln


class Bispherical(CurvedWall):
    def __init__(self, Ri: float, Ro: float, h: float,
                 t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = None, n_i: float = None, draw=False, trace=True, n_rays=21,
                 ray_y_range=None, x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        super().__init__(Ri, h, t, h_f, material, design_wavelength, n_o, n_i)
        self._Ro = Ro
        self.check_fix_geometry()
        if draw:
            self.draw()
        if trace:
            if ray_y_range is None:
                ray_y_range = self.default_Y_range
            self.trace_ray_set(design_wavelength, n_rays=n_rays, y_range=ray_y_range,
                               x_stop=x_stop, dx_extend=dx_extend, x_start=x_start, draw=draw_rays)

    @property
    def Ro(self) -> float:
        return self._Ro

    @property
    def t(self):
        return self.t_1

    @property
    def x_min(self):
        # return self.get_Qo('left')[0] - self.Ro
        return -self.x_max

    @property
    def x_max(self):
        if self.Ro > 0:
            return self.get_Qo('right')[0] + self.Ro
        else:
            return self.get_Qo('right')[0] - np.sqrt(self.Ro ** 2 - (self.h / 2) ** 2)

    @property
    def power(self):
        return optical_power(self.Ri, self.Ro, self.t, self.n_1, self.n_o, self.n_i)

    @property
    def default_optimization_parameters(self):
        return 't_1', 'power_1'

    @property
    def all_optimization_parameters(self):
        return 'Ro', 't_1', 'power_1'

    def set_parameter(self, p, name, preserve_power=True):
        P0 = self.power
        if name == 'Ro':
            self._Ro = p
        elif name == 't1' or name == 't_1':
            self._t_1 = p
            if preserve_power:
                self.set_Ro_by_power(P0)
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')

    def print_dims(self, prefix=''):
        print(prefix + f"Ri =\t{self.Ri:6.3f} {_length_unit}")
        print(prefix + f"Ro =\t{self.Ro:6.3f} {_length_unit}")
        print(prefix + f"t =\t{self.t_1:6.3f} {_length_unit}")
        print(prefix + f"power =\t{self.power:4.3f} {_power_unit}")

    def get_Qo(self, side):
        Qo_right = self.Qi + np.array(
            [-self.Ro + self.Ri + self.t, 0])  # center coordinates of the outer surface; defined for left window
        if side == 'right':
            return Qo_right
        elif side == 'left':
            return Qo_right * np.array([-1, 0])

    def ray_trace_exterior(self, ray: raytrace.Ray, side, diagnostics=False):
        print_diag('  Tracing the outer surface', diagnostics)
        n1 = self.get_n1(ray.wavelength)
        if side not in ['left', 'right']:
            pass
        elif side == 'left':
            ray.refract_circle(self.Ro, self.get_Qo(side), n1, Y_max=self.Y_max_o, diagnostics=diagnostics)
        elif side == 'right':
            ray.refract_circle(self.Ro, self.get_Qo(side), self.n_o, Y_max=self.Y_max_o, diagnostics=diagnostics)

    def get_xy_i(self, n_pts=100):
        # points on inner surface
        rad_ext_i = math.asin((self.h / 2 - self.h_f) / self.Ri)
        rad_i = np.linspace(-rad_ext_i, rad_ext_i, n_pts).reshape(-1)
        xi = np.array(self.Ri * np.cos(rad_i))
        yi = np.array(self.Ri * np.sin(rad_i))
        return xi, yi

    def get_xy_o(self, n_pts=100):
        # points on outer surface
        rad_ext_o = math.asin(self.h / 2 / self.Ro)
        rad_o = np.linspace(-rad_ext_o, rad_ext_o, n_pts).reshape(-1)
        xo = np.array(self.Ro * np.cos(rad_o) - self.get_Qo('left')[0])
        yo = np.array(self.Ro * np.sin(rad_o))
        return xo, yo

    def check_fix_geometry(self):
        xi, _ = self.get_xy_i(2)
        xo, _ = self.get_xy_o(2)
        dx = xo[0] - xi[0]
        if dx < 0:
            self._t_1 += np.abs(dx)

    def set_Ro_by_power(self, P: float):
        """
        Set the outer element radius as needed to be of a given optical power
        :param P: float, optical power, 1/f
        :return: None
        """
        self._Ro = Ro_specified_Power(P, self.Ri, self.t, self.n_1, self.n_o, self.n_i)

    @staticmethod
    def get_xy_m(_):
        # points on mid surface; undefined except for cemented doublets
        return 0, None

    def draw(self, which='both', n_pts=100, center_circle=False, ls='-', c='k', lw=2,
             new_figure=True, figsize=(10, 5)):

        xi, yi = self.get_xy_i(n_pts)
        xo, yo = self.get_xy_o(n_pts)
        xm, ym = self.get_xy_m(n_pts)

        if new_figure:
            plt.figure(figsize=figsize)
            plt.axis('equal')
        if center_circle:
            circle = plt.Circle((0, 0), self.Ri, ec=c, ls=':', lw=lw * .75, fill=False)
            plt.gca().add_patch(circle)

        if which in ['left', 'both']:
            self._draw_window(-xi, yi, -xo, yo, -xm, ym, ls=ls, c=c, lw=lw)
        if which in ['right', 'both']:
            self._draw_window(xi, yi, xo, yo, xm, ym, ls=ls, c=c, lw=lw)

    def draw_rays(self, sets='all', c='r', ls='-', marker='', arrow_size=8):
        for ray in self.rays.rays_of_set(sets):
            ray.draw(c=c, ls=ls, marker=marker, arrow_size=arrow_size)

    @staticmethod
    def _draw_window(xi, yi, xo, yo, xm=None, ym=None, ls='-', c='k', lw=2):
        plt.plot(xi, yi, c=c, ls=ls, lw=lw)
        plt.plot(xo, yo, c=c, ls=ls, lw=lw)
        if xm is not None and ym is not None:
            plt.plot(xm, ym, c=c, ls=ls, lw=lw)
        plt.plot([xo[0], xi[0], xi[0]], [yo[0], yo[0], yi[0]], c=c, ls=ls, lw=lw)
        plt.plot([xo[-1], xi[-1], xi[-1]], [yo[-1], yo[-1], yi[-1]], c=c, ls=ls, lw=lw)
        plt.axis('equal')


class Concentric(Bispherical):

    def __init__(self, Ri: float, h: float, t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = None, n_i: float = None, draw=False, trace=True, n_rays=21, ray_y_range=None,
                 x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        Ro = Ri + t
        super().__init__(Ri, Ro, h, t, h_f=h_f, material=material, design_wavelength=design_wavelength,
                         n_o=n_o, n_i=n_i, draw=draw, trace=trace, n_rays=n_rays, ray_y_range=ray_y_range,
                         x_start=x_start, x_stop=x_stop, dx_extend=dx_extend, draw_rays=draw_rays)

    @property
    def Ro(self):
        return self.Ri + self.t

    def draw_annulus(self, c='k', ls='-', lw=1.5):
        for R in self.Ri, self.Ro:
            plt.gca().add_patch(plt.Circle((0, 0), R, ec=c, ls=ls, lw=lw, fill=False))
        plt.axis('equal')


class ZeroPowerSinglet(Bispherical):
    def __init__(self, Ri: float, h: float, t: float, Ro='analytical', h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: float = 532, n_o: float = None, n_i: float = None, draw=False, trace=True,
                 n_rays=21, ray_y_range=None, x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        n1 = refractive_index(material, design_wavelength)
        n_o, n_i = check_no_ni(n_o, n_i)
        metric = None
        if isinstance(Ro, str):
            if Ro == 'analytical':
                Ro = ZeroPowerSinglet.Ro_analytical(Ri, t, n1, n_i=n_i, n_o=n_o)
            elif Ro in ['max_aberration', 'rms_aberration']:
                metric, Ro = Ro, Ri + t
            else:
                raise ValueError(f"Ro argument must be numeric radius or one of 'analytical', 'max_aberration',"
                                 f" or 'rms_aberration'.")

        super().__init__(Ri, Ro, h, t, h_f=h_f, material=material, design_wavelength=design_wavelength,
                         n_o=n_o, n_i=n_i, draw=draw, trace=trace, n_rays=n_rays, ray_y_range=ray_y_range,
                         x_start=x_start, x_stop=x_stop, dx_extend=dx_extend, draw_rays=draw_rays)

        if metric in ['max_aberration', 'rms_aberration']:
            self.optimize_Ro(metric=Ro, inplace=True, n_rays=n_rays, y_range=ray_y_range)

    @classmethod
    def Ro_analytical(cls, Ri, t, n1, n_i=1., n_o=1.):
        # return (n1 - 1) * (n1 * Ri - (n_o - n1) * t) / ((n1 - n_o) * n1)  # Vikram form
        return Ri * ((n1 - n_o) / (n1 - n_i)) * (n1 * Ri + (n1 - n_i) * t) / (n1 * Ri)

    @classmethod
    def Ro_rms_aberration(cls, Ri, t, n1, n_i=1., n_o=1.):
        # TODO: implement Ro based on minimum rms aberration
        return None

    @classmethod
    def Ro_max_aberration(cls, Ri, t, n1, n_i=1., n_o=1.):
        # TODO: implement Ro based on minimum rms aberration
        return None

    def optimize_Ro(self, metric, wavelength=None, inplace=True, n_rays=21, y_range=None,
                    Ro_guess=None, ray_sets='success'):
        err = None

        window = self.inplace(inplace)
        if Ro_guess is not None:
            window._Ro = Ro_guess

        if metric == 'analytical':
            Ro = ZeroPowerSinglet.Ro_analytical(window.Ri, window.t, window.n_1, window.n_i, window.n_o)
            window._Ro = Ro
            window.trace_ray_set(y_range=y_range, wavelength=wavelength, n_rays=n_rays,
                                 draw=False, inplace=True)
            return window

        if metric == 'rms_aberration':
            if Ro_guess is None:
                window._Ro = window.optimize_Ro('analytical', inplace=False).Ro

            def err(p):
                window._Ro = p[0]
                rays = window.trace_ray_set(y_range=y_range, wavelength=wavelength, n_rays=n_rays,
                                            draw=False, inplace=False)
                return rays.rms_aberration(sets=ray_sets)

        elif metric == 'max_aberration':
            print_diag('Solving as "max_aberration" problem.', False)
            if Ro_guess is None:
                window._Ro = (window.optimize_Ro('analytical', inplace=False).Ro +
                              window.optimize_Ro('rms_aberration', inplace=False).Ro) / 2

            def err(p):
                print_diag(p[0], False)
                window._Ro = p[0]
                rays = window.trace_ray_set(y_range=y_range, wavelength=wavelength, n_rays=n_rays,
                                            draw=False, inplace=False)
                return np.max(np.abs(rays.aberrations(sets=ray_sets)))

        else:
            ValueError(f"'metric' argument must be one of 'analytical', 'rms_aberration', or 'max_aberration'.")

        sln = minimize(err, np.array([window.Ro]), jac='2-point')

        window._Ro = sln.x[0]
        window.trace_ray_set(wavelength=wavelength, n_rays=n_rays, y_range=y_range, draw=False, inplace=True)

        return window


class CementedDoublet(Bispherical):

    # opt_params = ('Rm', 'Ro', 't_base', 't_crown')

    def __init__(self, Ri: float, Rm: float, Ro: float, h: float, t1: float, t2: float,
                 h_f: float = 0., wetted_material: str = 'fused silica', crown_material: str = 'bk7',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532, n_o: float = None, n_i: float = None,
                 draw=False, trace=True, n_rays=21, ray_y_range=None, x_start=None,
                 x_stop=None, dx_extend=None, draw_rays=False):
        self._Rm = float(Rm)
        self._t_2 = float(t2)
        self._material_2 = crown_material
        # self._opt_params = ('Rm', 'Ro', 't_base', 't_crown')
        super().__init__(Ri=Ri, Ro=Ro, h=h, t=t1, h_f=h_f, material=wetted_material,
                         design_wavelength=design_wavelength,
                         n_o=n_o, n_i=n_i, draw=draw, trace=trace, n_rays=n_rays, ray_y_range=ray_y_range,
                         x_start=x_start, x_stop=x_stop, dx_extend=dx_extend, draw_rays=draw_rays)

    @property
    def Rm(self):
        return self._Rm

    @property
    def t_2(self):
        return self._t_2

    @property
    def material_2(self):
        return self._material_2

    @property
    def n_2(self):
        design_wavelengths = self.design_wavelength
        if isinstance(design_wavelengths, typing.Iterable):
            return [refractive_index(self.material_2, wave) for wave in design_wavelengths]
        else:
            return refractive_index(self.material_2, design_wavelengths)

    @property
    def power_1(self):
        return optical_power(self.Ri, self.Rm, self.t_1, self.n_1, self.n_2, self.n_i)

    @property
    def power_2(self):
        return optical_power(self.Rm, self.Ro, self.t_2, self.n_2, self.n_o, self.n_1)

    @property
    def power(self):
        return None

    @property
    def optimization_metrics(self):
        return 'rms_aberration', 'max_aberration'

    @property
    def optimization_metric(self):
        return self._optimization_metric

    @optimization_metric.setter
    def optimization_metric(self, value):
        if value in self.optimization_metrics:
            self._optimization_metric = value
        else:
            raise ValueError(f'Metric must be one of "{self.optimization_metrics}"')

    @property
    def default_optimization_parameters(self):
        return 't_1', 't_2', 'power_1', 'power_2'

    @property
    def all_optimization_parameters(self):
        return 'Rm', 'Ro', 't_1', 't_2', 'power_1', 'power_2'

    def set_parameter(self, name, p, preserve_power=True, check_geometry=True):
        P1, P2 = self.power_1, self.power_2
        if name == 'Rm':
            self._Rm = p
        elif name == 'Ro':
            self._Ro = p
        elif name == 't1' or name == 't_1':
            self._t_1 = p
            if preserve_power:
                self.set_Rm_by_power_1(P1)
                self.set_Ro_by_power_2(P2)
        elif name == 't2' or name == 't_2':
            self._t_2 = p
            if preserve_power:
                self.set_Ro_by_power_2(P2)
        elif name == 'P1' or name == 'power_1':
            self.set_Rm_by_power_1(p)
            if preserve_power:
                self.set_Ro_by_power_2(P2)
        elif name == 'P2' or name == 'power_2':
            self.set_Ro_by_power_2(p)
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')
        self.check_fix_geometry()

    def get_parameter(self, name):
        if name == 'Rm':
            return self.Rm
        elif name == 'Ro':
            return self.Ro
        elif name == 't1' or name == 't_1':
            return self.t_1
        elif name == 't2' or name == 't_2':
            return self.t_2
        elif name == 'P1' or name == 'power_1':
            return self.power_1
        elif name == 'P2' or name == 'power_2':
            return self.power_2
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')

    def print_dims(self, prefix=''):
        print(prefix + f"Ri =      {self.Ri:6.3f} {_length_unit}")
        print(prefix + f"Rm =      {self.Rm:6.3f} {_length_unit}")
        print(prefix + f"Ro =      {self.Ro:6.3f} {_length_unit}")
        print(prefix + f"t_base =  {self.t_1:6.3f} {_length_unit}")
        print(prefix + f"t_crown = {self.t_2:6.3f} {_length_unit}")
        print(prefix + f"power_base =  {self.power_1:4.3f} {_power_unit}")
        print(prefix + f"power_crown = {self.power_2:4.3f} {_power_unit}")

    def get_n2(self, wavelength=None):
        if wavelength is None:
            return self.n_2
        return refractive_index(self.material_2, wavelength=wavelength)

    def get_Qo(self, side):
        # Qo_right = self.get_Qm('right') + np.array([-self.Ro + self.Rm + self.t_2, 0])
        Qo_right = np.array([-self.Ro + self.Ri + self.t_2 + self.t_1, 0])
        if side == 'right':
            return Qo_right
        elif side == 'left':
            return Qo_right * np.array([-1, 0])

    def get_Qm(self, side):
        Qm_right = self.Qi + np.array(
            [-self.Rm + self.Ri + self.t_1, 0])  # center coordinates of the outer surface; defined for left window
        if side == 'right':
            return Qm_right
        elif side == 'left':
            return Qm_right * np.array([-1, 0])

    def get_xy_m(self, n_pts=100):
        # points on middle surface
        if self.Rm != np.inf:
            rad_ext_m = math.asin(self.h / 2 / self.Rm)
            rad_m = np.linspace(-rad_ext_m, rad_ext_m, n_pts).reshape(-1)
            xm = self.Rm * np.cos(rad_m) - self.get_Qm('left')[0]
            ym = self.Rm * np.sin(rad_m)
        else:
            xm = np.array(2 * [self.Qi[0] - self.Ri - self.t_1])
            ym = np.array([1, -1]) * (self.h / 2)
        return xm, ym

    def ray_trace_exterior(self, ray: raytrace.Ray, side, diagnostics=False):
        print_diag('  Tracing the outer surface', diagnostics)
        n2 = self.get_n2(ray.wavelength)
        if side not in ['left', 'right']:
            pass
        elif side == 'left':
            ray.refract_circle(self.Ro, self.get_Qo(side), n2, Y_max=self.Y_max_o, diagnostics=diagnostics)
        elif side == 'right':
            ray.refract_circle(self.Ro, self.get_Qo(side), self.n_o, Y_max=self.Y_max_o, diagnostics=diagnostics)

    def ray_trace_middle(self, ray: raytrace.Ray, side='left', diagnostics=False):
        print_diag('  Tracing the middle surface', diagnostics)
        if side not in ['left', 'right']:
            pass
        elif side == 'left':
            n1 = self.get_n1(ray.wavelength)
            if self.Rm != np.inf:
                ray.refract_circle(self.Rm, self.get_Qm(side), n1, Y_max=self.Y_max_o, diagnostics=diagnostics)
            else:
                x = self.Qi[0] - self.Ri - self.t_1
                ray.refract_plane(x, n1, Y_max=self.Y_max_o, inplace=True)
        elif side == 'right':
            n2 = self.get_n2(ray.wavelength)
            if self.Rm != np.inf:
                ray.refract_circle(self.Rm, self.get_Qm(side), n2, Y_max=self.Y_max_o, diagnostics=diagnostics)
            else:
                x = self.Qi[0] + self.Ri + self.t_1
                ray.refract_plane(x, n2, Y_max=self.Y_max_o, inplace=True)
        pass

    def check_fix_geometry(self):
        f_rad = 1.2
        # first, check for valid radii
        if np.abs(self.Rm) < self.h / 2:
            self._Rm = f_rad * self.h / 2 * np.sign(self.Rm)
        if np.abs(self.Ro) < self.h / 2:
            self._Ro = f_rad * self.h / 2 * np.sign(self.Ro)
        # then, check valid thicknesses
        xi, _ = self.get_xy_i(2)
        xm, _ = self.get_xy_m(2)
        xo, _ = self.get_xy_o(2)
        dx1 = xm[0] - xi[0]
        dx2 = xo[0] - xm[0]
        if dx1 < 0:
            self._t_1 += np.abs(dx1)
        if dx2 < 0:
            self._t_2 += np.abs(dx2)

    def set_Rm_by_power_1(self, P: float):
        """
        Set the outer element radius as needed to be of a given optical power
        :param P: float, optical power, 1/f
        :return: None
        """
        self._Ro = Ro_specified_Power(P, self.Ri, self.t_1, self.n_1, self.n_2, self.n_i)

    def set_Ro_by_power_2(self, P: float):
        """
        Set the outer element radius as needed to be of a given optical power
        :param P: float, optical power, 1/f
        :return: None
        """
        self._Ro = Ro_specified_Power(P, self.Rm, self.t_2, self.n_2, self.n_o, self.n_1)

    def set_Ro_by_power(self, P: float):
        raise NotImplementedError('"set_Ro_by_power" not implemented for doublet windows. ' +
                                  'See "set_Rm_by_power_1" and "set_Ro_by_power_2" instead.')

class BisphericalLens(Bispherical):
    def __init__(self, Ri: float, Ro: float, h: float, t: float, xi: float,
                 h_f: float = 0., material: str = 'fused silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = 1., n_i: float = 1., draw=False, trace=False, n_rays=21,
                 ray_y_range=None, x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        self._xi = xi
        super().__init__(Ri, Ro, h, t, h_f, material, design_wavelength, n_o, n_i, trace=False)
        if draw:
            self.draw()
        if trace:
            if ray_y_range is None:
                ray_y_range = self.default_Y_range
            self.trace_ray_set(design_wavelength, n_rays=n_rays, y_range=ray_y_range,
                               x_stop=x_stop, dx_extend=dx_extend, x_start=x_start, draw=draw_rays)

    @property
    def Qi(self):
        raise ValueError('Property "Qi" undefined for BisphericalLens class')

    @property
    def xi(self):
        return self._xi

    def get_Qi(self, side):
        Qi_right = np.array([np.abs(self.xi) - self.Ri, 0])  # center coordinates of inner surface
        if side == 'right':
            return Qi_right
        elif side == 'left':
            return Qi_right * np.array([-1, 0])

    def get_Qo(self, side):
        Qo_right = np.array([np.abs(self.xi) + self.t - self.Ro, 0])  # center coordinates of the outer surface
        if side == 'right':
            return Qo_right
        elif side == 'left':
            return Qo_right * np.array([-1, 0])

    def ray_trace_interior(self, ray: raytrace.Ray, side: str = 'left', diagnostics=False):
        print_diag('  Tracing the inner surface', diagnostics)
        n1 = self.get_n1(ray.wavelength)
        if side not in ['left', 'right']:
            raise ValueError('"side" must be one of "left" or "right"')
        elif self.Ri == np.inf:
            n_new, sign = {'left': (self.n_o, -1), 'right': (self.n_1, 1)}[side]
            x = sign * (self.xi + self.Ri)
            ray.refract_plane(x, n_new, Y_max=self.Y_max_i)
        elif side == 'left':
            ray.refract_circle(self.Ri, self.get_Qi(side), self.n_o, Y_max=self.Y_max_i, diagnostics=diagnostics)
        elif side == 'right':
            ray.refract_circle(self.Ri, self.get_Qi(side), n1, Y_max=self.Y_max_i, diagnostics=diagnostics)

    def get_xy_i(self, n_pts=100):
        # points on inner surface
        rad_ext_i = math.asin((self.h / 2 - self.h_f) / self.Ri)
        rad_i = np.linspace(-rad_ext_i, rad_ext_i, n_pts).reshape(-1)
        xi = self.Ri * np.cos(rad_i) - self.get_Qi('left')[0]
        yi = self.Ri * np.sin(rad_i)
        return xi, yi

    def get_xy_o(self, n_pts=100):
        # points on outer surface
        rad_ext_o = math.asin(self.h / 2 / self.Ro)
        rad_o = np.linspace(-rad_ext_o, rad_ext_o, n_pts).reshape(-1)
        xo = self.Ro * np.cos(rad_o) - self.get_Qo('left')[0]
        yo = self.Ro * np.sin(rad_o)
        return xo, yo

    # def draw(self, which='both', n_pts=100, center_circle=False, ls='-', c='k', lw=2,
    #          new_figure=True, figsize=(10, 5)):
    #     xi, yi = self.get_xy_i(n_pts)
    #
    #     xo, yo = self.get_xy_o(n_pts)
    #
    #     # points on mid surface
    #     xm, ym = self.get_xy_m(n_pts)
    #
    #     if new_figure:
    #         plt.figure(figsize=figsize)
    #         plt.axis('equal')
    #     if center_circle:
    #         circle = plt.Circle((0, 0), self.Ri, ec=c, ls=':', lw=lw * .75, fill=False)
    #         plt.gca().add_patch(circle)
    #
    #     if which in ['left', 'both']:
    #         _draw_window(-xi, yi, -xo, yo, -xm, ym, ls=ls, c=c, lw=lw)
    #     if which in ['right', 'both']:
    #         _draw_window(xi, yi, xo, yo, xm, ym, ls=ls, c=c, lw=lw)

    def check_fix_geometry(self):
        xi, _ = self.get_xy_i(2)
        xo, _ = self.get_xy_o(2)
        dx = xo[0] - xi[0]
        if dx < 0:
            self._t_1 += np.abs(dx)


class AirspaceDoublet(CurvedWall):

    opt_params = ('Ro', 't_base', 'R_lens_i', 'R_lens_o', 't_lens', 'x_air')

    def __init__(self, window: CurvedWall, lens: BisphericalLens,
                 design_wavelength: typing.Union[float, typing.Iterable] = None):

        if design_wavelength is None:
            design_wavelength = window.design_wavelength
        window._design_wavelength = design_wavelength
        lens._design_wavelength = design_wavelength
        super().__init__(0., 0., 0., 0., design_wavelength=design_wavelength)
        self._window = window
        self._lens = lens

    @property
    def window(self):
        return self._window

    @property
    def lens(self):
        return self._lens

    @property
    def x_min(self):
        return self.lens.x_min

    @property
    def x_max(self):
        return self.lens.x_max

    @property
    def x_airgap(self):
        return self.lens.xi - self.window.x_max

    @property
    def estimated_Y_max_in(self):
        wave = self.design_wavelength
        if isinstance(wave, typing.Iterable):
            wave = list(wave)[0]
        ray = raytrace.Ray([0, self.window.h / 2. - self.window.h_f], wave, self.window.n_i)
        self.ray_trace_window(ray, 'right')
        return ray.Pr[1]

    @property
    def default_bounds(self):
        return {'Ri_lens': (-1.8 / self.lens.Y_max_i, 1.8 / self.lens.Y_max_i),
                'Ro_lens': (-1.8 / self.lens.Y_max_o, 1.8 / self.lens.Y_max_o),
                't_window': {self.window.h / 20., None},
                't_lens': (self.lens.h / 20., None),
                'x_air': (self.window.t_1 / 10, None)}

    @property
    def optimization_metrics(self):
        return 'rms_aberration', 'max_aberration'

    @property
    def optimization_metric(self):
        return self._optimization_metric

    @optimization_metric.setter
    def optimization_metric(self, value):
        if value in self.optimization_metrics:
            self._optimization_metric = value
        else:
            raise ValueError(f'Metric must be one of "{self.optimization_metrics}"')

    @property
    def default_optimization_parameters(self):
        return 't_lens', 'Ri_lens', 'power_lens', 'x_air'

    @property
    def all_optimization_parameters(self):
        return 't_window', 't_lens', 'Ri_lens', 'Ro_lens', 'x_air', 'power_lens'

    def set_parameter(self, name, p, preserve_power=True, check_geometry=True):
        P_lens = self.lens.power
        if name == 't_window':
            self.window._t_1 = p
        elif name == 't_lens':
            self.lens._t_1 = p
            if preserve_power:
                self.lens.set_Ro_by_power(P=P_lens)
        elif name == 'Ri_lens':
            self._Ri = p
        elif name == 'Ro_lens':
            self.lens._Ro = p
        elif name == 'x_air':
            self.lens._xi = p + self.window.x_max
        elif name == 'P_lens' or name == 'power_lens':
            self.lens.set_Ro_by_power(p)
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')
        self.check_fix_geometry()

    def get_parameter(self, name):
        if name == 't_window':
            return self.window.t_1
        elif name == 't_lens':
            return self.lens.t
        elif name == 'Ri_lens':
            return self.lens.Ri
        elif name == 'Ro_lens':
            return self.lens.Ro
        elif name == 'x_air':
            return self.x_airgap
        elif name == 'P_lens' or name == 'power_lens':
            return self.lens.power
        else:
            ValueError(f'Parameter name "{name}" invalid for window of type {type(self)}')

    def draw(self, which='both', n_pts=100, center_circle=True, ls='-', c='k', lw=2, new_figure=True, figsize=(10, 5)):

        self.lens.draw(which=which, n_pts=n_pts, center_circle=False, ls=ls, c=c, lw=lw, new_figure=new_figure,
                       figsize=figsize)
        if isinstance(self.window, Concentric):
            self.window.draw_annulus(ls=':')
            # plt.ylim(self.window.Ri * np.array([-1.1, 1.1]))
        self.window.draw(which=which, n_pts=n_pts, center_circle=center_circle, ls=ls, c=c, lw=lw, new_figure=False)

    def ray_trace_functions(self, side: str = 'left'):
        funcs = self.lens.ray_trace_window, self.window.ray_trace_window
        if side == 'left':
            return funcs
        elif side == 'right':
            return funcs[::-1]

    def print_dims(self, prefix: str = ''):
        print_md(f'_{str(type(self.window)).split(".")[-1][:-2]} Window:_')
        self.window.print_dims(prefix)
        print_md(f'_{str(type(self.lens)).split(".")[-1][:-2].split("Lens")[0]} Lens:_')
        self.lens.print_dims(prefix)
        print('\n' + f'x_airgap =\t{self.x_airgap:6.3f} {_length_unit}')

    # def optimize(self, metric='rms_aberration', which=4 * (True,), wavelength=None, inplace=True,
    #              n_rays=11, y_range=None, t1_bounds=None, t2_bounds=None, ray_sets='success'):
    #
    #     err = None
    #     window = self.inplace(inplace)
    #     which = np.array(which, dtype=np.bool)
    #     print_diag(which, False)
    #
    #     '''
    #     Define the optimization (error) function based on array 'p' of parameters, where:
    #         p[0] = 1 / Rm  # curvature of mid surface to provide continuity between positive and negative values
    #         p[1] = Ro  # outer surface radius
    #         p[2] = t1  # wetted element thickness
    #         p[3] = t2  # outer element thickness
    #     '''
    #
    #     def set_params(window_, p):
    #         i = 0
    #         if which[0]:
    #             if p[i] != 0:
    #                 window_._Rm = 2 / p[i]
    #             else:
    #                 window_._Rm = np.inf
    #             i += 1
    #         if which[1]:
    #             window_._Ro = p[i]
    #             i += 1
    #         if which[2]:
    #             window_._t_1 = p[i]
    #             i += 1
    #         if which[3]:
    #             window_._t_2 = p[i]
    #         try:
    #             self.check_fix_geometry()
    #         except ValueError:
    #             print(self.Rm, self.h)
    #             raise Exception()
    #
    #     if y_range is None:
    #         y_range = self.default_Y_range
    #
    #     if metric == 'rms_aberration':
    #         def err(p):
    #             set_params(window, p)
    #             rays = window.trace_ray_set(y_range=y_range, wavelength=wavelength, n_rays=n_rays,
    #                                         draw=False, inplace=False)
    #             return rays.rms_aberration(sets=ray_sets)
    #
    #     elif metric == 'max_aberration':
    #         def err(p):
    #             print_diag(p[0], False)
    #             set_params(window, p)
    #             rays = window.trace_ray_set(y_range=y_range, wavelength=wavelength, n_rays=n_rays,
    #                                         draw=False, inplace=False)
    #             return np.max(np.abs(rays.aberrations(sets=ray_sets)))
    #
    #     else:
    #         ValueError(f"'metric' argument must be one of 'rms_aberration' or 'max_aberration'.")
    #
    #     if t1_bounds is None:
    #         t1_bounds = (0.03 * self.h, 0.5 * self.h)
    #     if t2_bounds is None:
    #         t2_bounds = (0.03 * self.h, 0.5 * self.h)
    #
    #     p0 = np.array([2 / window.Rm, window.Ro, window.t_1, window.t_2])[which]
    #     bounds = [bnd for bnd, w in zip([(-1.8 / self.Y_max_o, 1.8 / self.Y_max_o), (self.Y_max_o, None),
    #                                      t1_bounds, t2_bounds], which) if w]
    #
    #     print_diag(f'p0 =    \t{p0}\nbounds =\t{bounds}', False)
    #     print_diag(f'error0 =\t{err(p=p0)}', False)
    #
    #     sln = minimize(err, p0, jac='2-point', bounds=bounds)
    #
    #     set_params(window, sln.x)
    #     window.trace_ray_set(wavelength=wavelength, n_rays=n_rays, y_range=y_range, draw=False, inplace=True)
    #
    #     return window

    def check_fix_geometry(self):
        self.window.check_fix_geometry()
        self.lens.check_fix_geometry()


def singlet_Ro_analytical(Ri: float, t: float, n1: float,
                          n_i: float = 1., n_o: float = 1.):
    """
    Calculate the required outer window radius for a fixed inner radius and specified thickness and refractive index
    :param t: center thickness
    :return:
    """
    # return (n1 - 1) * (n1 * Ri - (n_o - n1) * t) / ((n1 - n_o) * n1)  # Vikram form
    return Ri * ((n1 - n_o) / (n1 - n_i)) * (n1 * Ri + (n1 - n_i) * t) / (n1 * Ri)


def optical_power(Ri: float, Ro: float, t: float, n: float, no: float = 1, ni: float = 1) -> float:
    """
    Calculate the optical power of a singlet element
    :param Ri: float, inner radius
    :param Ro: float, outer radius
    :param t: float, thickness
    :param n: float, element refractive index
    :param no: float, outer medium refractive index
    :param ni: float, inner medium refractive index
    :return: float
    """
    return ((n - no) / Ro + (ni - n) / Ri - (n - no) * (ni - n) * t / (n * Ri * Ro)) * power_mult()


def Ro_specified_Power(P: float, Ri: float, t: float, n: float, no: float = 1, ni: float = 1) -> float:
    """
    Calculates the outer radius of an optical element required to achieve a given optical power
    :param P: float, optical power, P = 1/f
    :param Ri: float, inner radius
    :param t: float, element thickness
    :param n: float, element refractive index
    :param no: float, outside medium refractive index
    :param ni: float, inside medium refractive index
    :return: Ro, float
    """
    P = P / power_mult()

    return ((n-no)*n*Ri + (n-no)*(n-ni)*t) / (n*(P*Ri+n-ni))