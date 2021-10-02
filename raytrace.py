import math
import typing
import copy
import numpy as np
import matplotlib.pyplot as plt

from utilities import print_diag, IntersectionExceedsBounds, NegativeIntersectionDistances


class Ray(object):
    def __init__(self, xy0, wavelength, n0, c='r', dxy_0=(1, 0), t_tol=1e-3):
        self._Pr = (np.array(xy0).reshape(-1),)
        self._Vr = (np.array(dxy_0).reshape(-1),)
        self._n = (n0,)
        self._wavelength = wavelength
        self._t_tol = t_tol
        self._success = True
        self.color = c

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def Pr(self):
        return self._Pr[-1]

    @property
    def Vr(self):
        return self._Vr[-1]

    @property
    def n(self):
        return self._n[-1]

    @property
    def xs(self):
        return np.array(self._Pr).T[0]

    @property
    def ys(self):
        return np.array(self._Pr).T[1]

    @property
    def Vrs(self):
        return self._Vr

    @property
    def success(self):
        return self._success

    def copy(self):
        return copy.deepcopy(self)

    def refract_circle(self, R, Q, n_new, Y_max=None, diagnostics=False, inplace=True):
        print_diag('    Trying to calculate the refraction of the ray...', diagnostics)
        Pr, Vr, valid = _spherical_refraction(self.Pr, self.Vr, Q, R, self.n, n_new,
                                              Y_max=Y_max, diag=diagnostics)
        if inplace:
            self._Pr += (Pr,)
            self._Vr += (Vr,)
            self._n += (n_new,)
            self._success *= valid

        return Pr, Vr, valid

    def refract_plane(self, x, n_new, Y_max=None, inplace=True):
        # Pr = self.extend_to_x(x, inplace=inplace)
        # valid = np.abs(Pr[1]) <= Y_max
        Pr, Vr, valid = _plano_refraction(self.Pr, self.Vr, x, self.n, n_new, Y_max=Y_max)
        if inplace:
            self._Pr += (Pr,)
            self._Vr += (Vr,)
            self._n += (n_new,)
            self._success *= valid

        return Pr, Vr, valid

    def extend_to_x(self, x, inplace=True):
        Pr = self.Pr
        dx = x - Pr[0]
        if dx > 0:
            return self.extend_dx(dx, inplace)

    def extend_dx(self, dx, inplace=True):
        # Pr = self.Pr
        Pr = self.Pr + dx * (self.Vr / self.Vr[0])
        if inplace:
            self._Vr += (self.Vr,)
            self._Pr += (Pr,)
        return Pr

    def draw(self, c=None, lw=1, ls='-', marker='', arrows=True, arrow_size=12):
        if c is None:
            c = self.color
        if ls is None:
            ls = {True: '-', False: ':'}[self._success]
        plt.plot(self.xs, self.ys, c=c, lw=lw, ls=ls, marker=marker)
        if arrows:
            arrow = [(-1, 0.5), (-0.5, 0), (-1, -0.5), (1, 0)]
            for i in [0, -1]:
                plt.plot(self.xs[i], self.ys[i], c=c, lw=lw, ls='', marker=arrow, mfc=c, ms=arrow_size)


class RaySet(object):
    def __init__(self):
        self._success = ()
        self._failed = ()

    @property
    def success(self):
        return self._success

    @property
    def failed(self):
        return self._failed

    def add_success(self, ray: Ray):
        self._success += ray,

    def add_failed(self, ray: Ray):
        self._failed += ray,

    def add_ray(self, ray: Ray):
        if ray.success:
            self._success += ray,
        else:
            self._failed += ray,

    def rays_of_set(self, sets='all', wavelengths=None):
        rays = ()
        if sets in ['all', 'success']:
            rays += self._success
        if sets in ['all', 'failed']:
            rays += self._failed
        if wavelengths is None:
            return rays
        elif not isinstance(wavelengths, typing.Iterable):
            wavelengths = [wavelengths]
        return [r for r in rays if r.wavelength in wavelengths]

    def ys_at_index(self, i, sets='all'):
        return [ray.ys[i] for ray in self.rays_of_set(sets=sets)]

    def ys_in(self, sets='all'):
        return self.ys_at_index(0, sets=sets)

    def ys_out(self, sets='all'):
        return self.ys_at_index(-1, sets=sets)

    def angles_at_index(self, i, sets='all', unit='rad', wavelengths=None):
        Vrs = [ray.Vrs[i] for ray in self.rays_of_set(sets=sets, wavelengths=wavelengths)]
        unit_factor = {'rad': 1, 'deg': 180 / np.pi, 'arcmin': 180 * 60 / np.pi, 'arcsec': 180 * 60**2 / np.pi}[unit]
        return [(np.arctan(Vr[1] / Vr[0]) * unit_factor) for Vr in Vrs]

    def angles_in(self, sets='all', unit='rad', wavelengths=None):
        return self.angles_at_index(0, sets=sets, unit=unit, wavelengths=wavelengths)

    def angles_out(self, sets='all', unit='rad', wavelengths=None):
        return self.angles_at_index(-1, sets=sets, unit=unit, wavelengths=wavelengths)

    def aberrations(self, sets='all', unit='rad', wavelengths=None):
        ang_in = np.array(self.angles_in(sets=sets, unit=unit, wavelengths=wavelengths))
        ang_out = np.array(self.angles_out(sets=sets, unit=unit, wavelengths=wavelengths))
        return ang_in - ang_out

    def rms_aberration(self, sets='all', unit='rad', wavelengths=None):
        return np.sqrt(np.mean((self.aberrations(sets=sets, unit=unit, wavelengths=wavelengths)) ** 2))

    def max_aberration(self, sets='all', unit='rad', wavelengths=None):
        return np.max(np.abs(self.aberrations(sets=sets, unit=unit, wavelengths=wavelengths)))

    def draw(self, sets='all', c='r', ls='-', marker='', arrow_size=8):
        for ray in self.rays_of_set(sets):
            ray.draw(c=c, ls=ls, marker=marker, arrow_size=arrow_size)


def _spherical_refraction(Pr, Vr, Q, R, n1, n2, Y_max=None, diag=False):

    try:
        Pr_int, in_bounds = _spherical_intersection(Pr, Vr, Q, R, Y_max=Y_max), True
    except IntersectionExceedsBounds as err:
        Pr_int, in_bounds = err.Pr, False

    print_diag(f'{Pr}, {Pr_int}', diag)

    dc = Pr_int - Q  # x,y displacement from center to intersection point
    print_diag(dc, diag)

    a_surf = math.atan(dc[1]/dc[0])
    a_ray_i = math.atan(Vr[1] / Vr[0])

    da_ray_i = a_ray_i - a_surf
    da_ray_o = math.asin(n1 * math.sin(da_ray_i) / n2)

    a_ray_o = a_surf + da_ray_o

    V = np.array([1, math.tan(a_ray_o)])
    if V[0] < 0:
        V *= (-1, -1)

    return Pr_int, V, in_bounds


def _spherical_intersection(Pr, Vr, Q, R, Y_max=None, tol=1e-3):
    """
    Find intersection of line and circle
    adapted from: https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    # last point on line and direction

    # do some linear algebra
    a = Vr.dot(Vr)
    b = 2 * Vr.dot(Pr - Q)
    c = Pr.dot(Pr) + Q.dot(Q) - 2 * Pr.dot(Q) - R**2

    # determine if a solution exists
    disc = b**2 - 4 * a * c
    if disc < 0:
        raise Exception("Line does not intersect the circle")

    # try:  # TODO: delete old troubleshooting code
    #     if disc < 0:
    #         raise Exception("Line does not intersect the circle")
    # except:
    #     raise ValueError(f"Cannot evaluate {disc} < 0:"
    #                      f"\n\ta = {a}, b = {b}, c = {c}"
    #                      f"\n\tVr = {Vr}, Pr = {Pr}, Q = {Q}, R = {R}")

    # solve the system of equations
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    # determine which intersection is next
    if t1 > tol and t2 > tol:
        t = min([t1, t2])
    elif t1 > tol:
        t = t1
    elif t2 > tol:
        t = t2
    else:
        Prs = (Pr + t1 * Vr, Pr + t2 * Vr)
        raise NegativeIntersectionDistances(f'Error detecting next intersection:\n'
                                            f'\tDistances = ({t1}, {t2})\n'
                                            f'\tTolerance = {tol}', Prs=Prs)
        # raise Exception(f'Error detecting next intersection:\n'
        #                 f'\tDistances = ({t1}, {t2})\n'
        #                 f'\tTolerance = {tol}')

    P_int = Pr + t * Vr

    # check if the intercept occurs outside of the valid extent of the window
    if Y_max is not None:
        print_diag((P_int, Y_max), False)
        if abs(P_int[1]) > Y_max:
            # Pr += P_int
            raise IntersectionExceedsBounds(f'Intercept Y value {P_int[1]} greater than max value {Y_max}', Pr=P_int)

    return P_int


def _plano_refraction(Pr, Vr, x_lens, n1, n2, Y_max=None):
    """
    calulate the intersection point and refracted direction of a ray with a vertical plane
    :param Pr:
    :param Vr:
    :param x_lens:
    :param n1:
    :param n2:
    :return:
    """
    # calculate the plano surface intersection
    dx = x_lens - Pr[0]
    dy = dx / Vr[0] * Vr[1]
    Pr_int = Pr + (dx, dy)
    # print(f'Pr =\t{Pr}\nPr_int =\t{Pr_int}\ndx =\t{dx}\ndy =\t{dy}\nx_lens =\t{x_lens}')

    # calculate the transmitted ray angle
    ang_i = math.atan(Vr[1] / Vr[0])
    ang_o = math.asin(n1 * math.sin(ang_i) / n2)

    # calculate the transmitted ray direction vector
    V = np.array([1, math.tan(ang_o)])
    if Vr[0] < 0:
        V *= (-1, -1)
    Vr = V

    if Y_max is not None:
        valid = abs(Pr_int[1]) <= Y_max
    else:
        valid = True

    # print(f'Plano refraction result:\n\tPr =\t{Pr_int}\n\tVr =\t{Vr}\n\tvalid =\t{valid}')
    return Pr_int, Vr, valid
