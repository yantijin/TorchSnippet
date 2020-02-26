from .solvers import FixedGridODESolver


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


def rk4_step_func(func, t, dt, y, k1=None):
    '''
        `y_{n+1} = y_n + \\frac{\\Delta t}{6} . (k_1 + 2k_2 + 2k_3+k4)`
        `k1=f(t_n,y_n)`
        `k_2 = f(t_n+\\frac{\\Delta t}{2}, y_n+\\frac{\\Delta t}{2}*k_1)`
        `k_3 = f(t_n+\\frac{\\Delta t}{2}, y_n+\\frac{\\Delta t}{2}*k_2)`
        `k_4 = f(t_n+\\Delta t, y_n+\\Delta t*k_3)`
    '''
    if k1 is None: k1 = func(t, y)
    k2 = func(t + dt / 2, tuple(y_ + dt * k1_ / 2 for y_, k1_ in zip(y, k1)))
    k3 = func(t + dt / 2, tuple(y_ + dt * k2_ / 2 for y_, k2_ in zip(y, k2)))
    k4 = func(t + dt, tuple(y_ + dt * k3_ for y_, k3_ in zip(y, k3)))
    return tuple((k1_ + 2 * k2_ + 2 * k3_ + k4_) * (dt / 6) for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


def rk4_alt_step_func(func, t, dt, y, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None: k1 = func(t, y)
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)))
    k3 = func(t + dt * 2 / 3, tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_, k2_ in zip(y, k1, k2)))
    k4 = func(t + dt, tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)))
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8) for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4