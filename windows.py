import math
import typing
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utilities import print_diag, refractive_index
import raytrace


class CurvedWall(object):
    def __init__(self, Ri: float, h: float, t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[typing.SupportsFloat, typing.Iterable] = 532,
                 n_o: typing.SupportsFloat = 1, n_i: typing.SupportsFloat = 1):
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

    def inplace(self, inplace):
        return {False: copy.deepcopy(self)}.get(inplace, self)

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

    def ray_trace_window(self, ray: raytrace.Ray, side, diagnostics=False):
        if side not in ['left', 'right']:
            raise ValueError('"Side" must be one of "left" or "right".')
        else:
            print_diag(f'Tracing the {side} window', diagnostics)
            trace_functions = self.ray_trace_functions(side)
            for f in trace_functions:
                f(ray, side, diagnostics)

    def ray_trace_windows(self, ray: raytrace.Ray, x_stop=None, dx_extend=None, diagnostics=True):
        self.ray_trace_window(ray, 'left', diagnostics=diagnostics)
        self.ray_trace_window(ray, 'right', diagnostics=diagnostics)
        if x_stop is not None:
            ray.extend_to_x(x_stop)
        elif dx_extend is not None:
            ray.extend_dx(dx_extend)

    def trace_ray_set(self, wavelength=None, n_rays=21, y_range=None, x_start=None, x_stop=None, dx_extend=None,
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
            x_start = self.x_min - 0.1 * np.abs(self.x_min)

        if x_stop is None and dx_extend is None:
            x_stop = self.x_max + 0.1 * np.abs(self.x_max)

        if not isinstance(wavelength, typing.Iterable):
            wavelength = [wavelength]
        for wave in wavelength:
            for y in ys_in:
                ray = raytrace.Ray((x_start, y), wave, self.n_o, c=color)
                self.ray_trace_windows(ray, x_stop=x_stop, dx_extend=dx_extend, diagnostics=diagnostics)
                if draw:
                    ray.draw(ls=linestyle, marker=marker, arrow_size=arrow_size)
                ray_set.add_ray(ray)

        if inplace:
            self._rays = ray_set

        return ray_set


class Bispherical(CurvedWall):
    def __init__(self, Ri: float, Ro: float, h: float,
                 t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = 1., n_i: float = 1., draw=False, trace=True, n_rays=21,
                 ray_y_range=None, x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        super().__init__(Ri, h, t, h_f, material, design_wavelength, n_o, n_i)
        self._Ro = Ro
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
        return self.get_Qo('left')[0] - self.Ro

    @property
    def x_max(self):
        return self.get_Qo('right')[0] + self.Ro

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

    def check_fix_geometry(self):
        xi, _ = self.get_xy_i(2)
        xo, _ = self.get_xy_o(2)
        dx = xo[0] - xi[0]
        if dx < 0:
            self._t_1 += np.abs(dx)

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
    @property
    def Ro(self):
        return self.Ri + self.t

    def __init__(self, Ri: float, h: float, t: float, h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: typing.Union[float, typing.Iterable] = 532,
                 n_o: float = 1, n_i: float = 1, draw=False, trace=True, n_rays=21, ray_y_range=None,
                 x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        Ro = Ri + t
        super().__init__(Ri, Ro, h, t, h_f=h_f, material=material, design_wavelength=design_wavelength,
                         n_o=n_o, n_i=n_i, draw=draw, trace=trace, n_rays=n_rays, ray_y_range=ray_y_range,
                         x_start=x_start, x_stop=x_stop, dx_extend=dx_extend, draw_rays=draw_rays)

    def draw_annulus(self, c='k', ls='-', lw=1.5):
        for R in self.Ri, self.Ro:
            plt.gca().add_patch(plt.Circle((0, 0), R, ec=c, ls=ls, lw=lw, fill=False))
        plt.axis('equal')


class ZeroPowerSinglet(Bispherical):
    def __init__(self, Ri: float, h: float, t: float, Ro='analytical', h_f: float = 0., material: str = 'fused_silica',
                 design_wavelength: float = 532, n_o: float = 1, n_i: float = 1, draw=False, trace=True,
                 n_rays=21, ray_y_range=None, x_start=None, x_stop=None, dx_extend=None, draw_rays=False):
        n1 = refractive_index(material, design_wavelength)
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
