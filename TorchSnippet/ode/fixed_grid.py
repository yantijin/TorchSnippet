from .solvers import FixedGridODESolver
from .rk_common import *


class Euler(FixedGridODESolver):
    '''
    `y'(t) = f(t, y(t))`
    `y_{n+1}=y_{n} + \\Delta t .f(t_n, y_n)`
    '''

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):
    '''
    `y_{n+1} = y_{n} + \\Delta t .f(t_n+\\frac{1}{2}\\Delta t, y_n + \\frac{1}{2}\\Delta t. f(t_n, y_n))`
    '''

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4