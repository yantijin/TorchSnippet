import abc
import torch
import sys
import collections
from .misc import _assert_increasing, _handle_unused_kwargs, _scaled_dot_product, _is_iterable, _interp_fit,\
    _convert_to_tensor, _select_initial_step, _interp_evaluate, _compute_error_ratio, \
    _optimal_step_size, _has_converged,_is_finite
from .rk_common import (_ButcherTableau, rk4_alt_step_func, rk4_step_func,
                        _runge_kutta_step, _RungeKuttaState)


__all__ = [
    'AdaptiveStepsizeODESolver', 'FixedGridODESolver',

    # fixed grid methods
    'Euler', 'Midpoint', 'RK4',

    # adaptive methods
    'AdamsBashforth', 'AdamsBashforthMoulton', 'AdaptiveHeunSolver',
    'VariableCoefficientAdamsBashforth', 'Bosh3Solver', 'Dopri5Solver',
    'Tsit5Solver'
]


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, **unused_kwargs):
        '''
        :param func: numerical step integreation function
        :param y0: initial value of ODE
        :param atol: error parameters
        :param rtol: arror parameters
        '''
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        '''
        :param t: integrate time list
        :return: solutions for numerical integration
        '''
        _assert_increasing(t)
        solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)
        return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta


    def __init__(self, func, y0, step_size=None, grid_constructor=None, **unused_kwargs):
        '''
        :param func: numerical step integration function
        :param y0: initial value for integration,
        :param step_size: step size for numerical integration
        :param grid_constructor: a function that construct a grid
        '''
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))
            y0 = y1

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1

        return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))


#####################fixed_grid####################################

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


################## adaptive heun ###############################
_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=[1.],
    beta=[
        [1.],
    ],
    c_sol=[0.5, 0.5],
    c_error=[
        0.5,
        -0.5,
    ],
)

AH_C_MID = [
    0.5, 0.
]


def _interp_fit_adaptive_heun(y0, y1, k, dt, tableau=_ADAPTIVE_HEUN_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = dt.type_as(y0[0])
    y_mid = tuple(y0_ + _scaled_dot_product(dt, AH_C_MID, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _abs_square(x):
    return torch.mul(x, x)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors


class AdaptiveHeunSolver(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 1, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_heun_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_heun_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_ADAPTIVE_HEUN_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_adaptive_heun(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state


#############################bosh3############################
_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=[1/2, 3/4,  1.],
    beta=[
        [1/2],
        [0., 3/4],
        [2/9, 1/3, 4/9]
    ],
    c_sol=[2/9, 1/3, 4/9, 0.],
    c_error=[2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8],
)

BS_C_MID = [ 0., 0.5,  0., 0.  ]


def _interp_fit_bosh3(y0, y1, k, dt):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = dt.type_as(y0[0])
    y_mid = tuple(y0_ + _scaled_dot_product(dt, BS_C_MID, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)



class Bosh3Solver(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_bosh3_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_bosh3_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_BOGACKI_SHAMPINE_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_bosh3(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=3
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state


########################dopri5###############################
_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    beta=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_error=[
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.,
        ],
)

DPS_C_MID = [
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
]


def _interp_fit_dopri5(y0, y1, k, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = dt.type_as(y0[0])
    y_mid = tuple(y0_ + _scaled_dot_product(dt, DPS_C_MID, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)



class Dopri5Solver(AdaptiveStepsizeODESolver):

    def __init__(
            self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
            **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 4, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_dopri5_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_dopri5_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_dopri5(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state


#####################Tsit5Solver########################################
_TSITOURAS_TABLEAU = _ButcherTableau(
    alpha=[0.161, 0.327, 0.9, 0.9800255409045097, 1., 1.],
    beta=[
        [0.161],
        [-0.008480655492357, 0.3354806554923570],
        [2.897153057105494, -6.359448489975075, 4.362295432869581],
        [5.32586482843925895, -11.74888356406283, 7.495539342889836, -0.09249506636175525],
        [5.86145544294642038, -12.92096931784711, 8.159367898576159, -0.071584973281401006, -0.02826905039406838],
        [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
    ],
    c_sol=[0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0],
    c_error=[
        0.09646076681806523 - 0.001780011052226,
        0.01 - 0.000816434459657,
        0.4798896504144996 - -0.007880878010262,
        1.379008574103742 - 0.144711007173263,
        -3.290069515436081 - -0.582357165452555,
        2.324710524099774 - 0.458082105929187,
        -1 / 66,
    ],
)


def _interp_coeff_tsit5(t0, dt, eval_t):
    t = float((eval_t - t0) / dt)
    b1 = -1.0530884977290216 * t * (t - 1.3299890189751412) * (t**2 - 1.4364028541716351 * t + 0.7139816917074209)
    b2 = 0.1017 * t**2 * (t**2 - 2.1966568338249754 * t + 1.2949852507374631)
    b3 = 2.490627285651252793 * t**2 * (t**2 - 2.38535645472061657 * t + 1.57803468208092486)
    b4 = -16.54810288924490272 * (t - 1.21712927295533244) * (t - 0.61620406037800089) * t**2
    b5 = 47.37952196281928122 * (t - 1.203071208372362603) * (t - 0.658047292653547382) * t**2
    b6 = -34.87065786149660974 * (t - 1.2) * (t - 0.666666666666666667) * t**2
    b7 = 2.5 * (t - 1) * (t - 0.6) * t**2
    return [b1, b2, b3, b4, b5, b6, b7]


def _interp_eval_tsit5(t0, t1, k, eval_t):
    dt = t1 - t0
    y0 = tuple(k_[0] for k_ in k)
    interp_coeff = _interp_coeff_tsit5(t0, dt, eval_t)
    y_t = tuple(y0_ + _scaled_dot_product(dt, interp_coeff, k_) for y0_, k_ in zip(y0, k))
    return y_t


def _optimal_rk_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next Runge-Kutta step."""
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=torch.float64, device=mean_error_ratio.device)
    error_ratio = torch.sqrt(mean_error_ratio).type_as(last_step)
    exponent = torch.tensor(1 / order).type_as(last_step)
    factor = torch.max(1 / ifactor, torch.min(error_ratio**exponent / safety, 1 / dfactor))
    return last_step / factor


class Tsit5Solver(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 4, self.rtol, self.atol).to(t)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(
            self.y0,
            self.func(t[0].type_as(self.y0[0]), self.y0), t[0], t[0], first_step,
            tuple(map(lambda x: [x] * 7, self.y0))
        )

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_tsit5_step(self.rk_state)
            n_steps += 1
        return _interp_eval_tsit5(self.rk_state.t0, self.rk_state.t1, self.rk_state.interp_coeff, next_t)

    def _adaptive_tsit5_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, _ = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_TSITOURAS_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_tol = tuple(self.atol + self.rtol * torch.max(torch.abs(y0_), torch.abs(y1_)) for y0_, y1_ in zip(y0, y1))
        tensor_error_ratio = tuple(y1_error_ / error_tol_ for y1_error_, error_tol_ in zip(y1_error, error_tol))
        sq_error_ratio = tuple(
            torch.mul(tensor_error_ratio_, tensor_error_ratio_) for tensor_error_ratio_ in tensor_error_ratio
        )
        mean_error_ratio = (
            sum(torch.sum(sq_error_ratio_) for sq_error_ratio_ in sq_error_ratio) /
            sum(sq_error_ratio_.numel() for sq_error_ratio_ in sq_error_ratio)
        )
        accept_step = mean_error_ratio <= 1

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        dt_next = _optimal_rk_step_size(dt, mean_error_ratio, self.safety, self.ifactor, self.dfactor)
        k_next = k if accept_step else self.rk_state.interp_coeff
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, k_next)
        return rk_state


######################adams#########################
_MIN_ORDER = 1
_MAX_ORDER = 12

gamma_star = [
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
]


class _VCABMState(collections.namedtuple('_VCABMState', 'y_n, prev_f, prev_t, next_t, phi, order')):
    """Saved state of the variable step size Adams-Bashforth-Moulton solver as described in

        Solving Ordinary Differential Equations I - Nonstiff Problems III.5
        by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett.
    """


def g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
    curr_t = prev_t[0]
    dt = next_t - prev_t[0]

    g = torch.empty(k + 1).to(prev_t[0])
    explicit_phi = collections.deque(maxlen=k)
    beta = torch.tensor(1).to(prev_t[0])

    g[0] = 1
    c = 1 / torch.arange(1, k + 2).to(prev_t[0])
    explicit_phi.append(implicit_phi[0])

    for j in range(1, k):
        beta = (next_t - prev_t[j - 1]) / (curr_t - prev_t[j]) * beta
        beat_cast = beta.to(implicit_phi[j][0])
        explicit_phi.append(tuple(iphi_ * beat_cast for iphi_ in implicit_phi[j]))

        c = c[:-1] - c[1:] if j == 1 else c[:-1] - c[1:] * dt / (next_t - prev_t[j - 1])
        g[j] = c[0]

    c = c[:-1] - c[1:] * dt / (next_t - prev_t[k - 1])
    g[k] = c[0]

    return g, explicit_phi


def compute_implicit_phi(explicit_phi, f_n, k):
    k = min(len(explicit_phi) + 1, k)
    implicit_phi = collections.deque(maxlen=k)
    implicit_phi.append(f_n)
    for j in range(1, k):
        implicit_phi.append(tuple(iphi_ - ephi_ for iphi_, ephi_ in zip(implicit_phi[j - 1], explicit_phi[j - 1])))
    return implicit_phi


class VariableCoefficientAdamsBashforth(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, implicit=True, first_step=None, max_order=_MAX_ORDER, safety=0.9, ifactor=10.0, dfactor=0.2,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.implicit = implicit
        self.first_step = first_step
        self.max_order = int(max(_MIN_ORDER, min(max_order, _MAX_ORDER)))
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)

    def before_integrate(self, t):
        prev_f = collections.deque(maxlen=self.max_order + 1)
        prev_t = collections.deque(maxlen=self.max_order + 1)
        phi = collections.deque(maxlen=self.max_order)

        t0 = t[0]
        f0 = self.func(t0.type_as(self.y0[0]), self.y0)
        prev_t.appendleft(t0)
        prev_f.appendleft(f0)
        phi.appendleft(f0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0).to(t)

        self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] + first_step, phi=phi, order=1)

    def advance(self, final_t):
        final_t = _convert_to_tensor(final_t).to(self.vcabm_state.prev_t[0])
        while final_t > self.vcabm_state.prev_t[0]:
            self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)
        assert final_t == self.vcabm_state.prev_t[0]
        return self.vcabm_state.y_n

    def _adaptive_adams_step(self, vcabm_state, final_t):
        y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
        if next_t > final_t:
            next_t = final_t
        dt = (next_t - prev_t[0])
        dt_cast = dt.to(y0[0])

        # Explicit predictor step.
        g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
        g = g.to(y0[0])
        p_next = tuple(
            y0_ + _scaled_dot_product(dt_cast, g[:max(1, order - 1)], phi_[:max(1, order - 1)])
            for y0_, phi_ in zip(y0, tuple(zip(*phi)))
        )

        # Update phi to implicit.
        next_f0 = self.func(next_t.to(p_next[0]), p_next)
        implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)

        # Implicit corrector step.
        y_next = tuple(
            p_next_ + dt_cast * g[order - 1] * iphi_ for p_next_, iphi_ in zip(p_next, implicit_phi_p[order - 1])
        )

        # Error estimation.
        tolerance = tuple(
            atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(y1_))
            for atol_, rtol_, y0_, y1_ in zip(self.atol, self.rtol, y0, y_next)
        )
        local_error = tuple(dt_cast * (g[order] - g[order - 1]) * iphi_ for iphi_ in implicit_phi_p[order])
        error_k = _compute_error_ratio(local_error, tolerance)
        accept_step = (torch.tensor(error_k) <= 1).all()

        if not accept_step:
            # Retry with adjusted step size if step is rejected.
            dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor, self.dfactor, order=order)
            return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order=order)

        # We accept the step. Evaluate f and update phi.
        next_f0 = self.func(next_t.to(p_next[0]), y_next)
        implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)

        next_order = order

        if len(prev_t) <= 4 or order < 3:
            next_order = min(order + 1, 3, self.max_order)
        else:
            error_km1 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 1] - g[order - 2]) * iphi_ for iphi_ in implicit_phi_p[order - 1]), tolerance
            )
            error_km2 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 2] - g[order - 3]) * iphi_ for iphi_ in implicit_phi_p[order - 2]), tolerance
            )
            if min(error_km1 + error_km2) < max(error_k):
                next_order = order - 1
            elif order < self.max_order:
                error_kp1 = _compute_error_ratio(
                    tuple(dt_cast * gamma_star[order] * iphi_ for iphi_ in implicit_phi_p[order]), tolerance
                )
                if max(error_kp1) < max(error_k):
                    next_order = order + 1

        # Keep step size constant if increasing order. Else use adaptive step size.
        dt_next = dt if next_order > order else _optimal_step_size(
            dt, error_k, self.safety, self.ifactor, self.dfactor, order=order + 1
        )

        prev_f.appendleft(next_f0)
        prev_t.appendleft(next_t)
        return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, order=next_order)


##########fixed_adams####################################
_BASHFORTH_COEFFICIENTS = [
    [],  # order 0
    [11],
    [3, -1],
    [23, -16, 5],
    [55, -59, 37, -9],
    [1901, -2774, 2616, -1274, 251],
    [4277, -7923, 9982, -7298, 2877, -475],
    [198721, -447288, 705549, -688256, 407139, -134472, 19087],
    [434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799],
    [14097247, -43125206, 95476786, -139855262, 137968480, -91172642, 38833486, -9664106, 1070017],
    [30277247, -104995189, 265932680, -454661776, 538363838, -444772162, 252618224, -94307320, 20884811, -2082753],
    [
        2132509567, -8271795124, 23591063805, -46113029016, 63716378958, -63176201472, 44857168434, -22329634920,
        7417904451, -1479574348, 134211265
    ],
    [
        4527766399, -19433810163, 61633227185, -135579356757, 214139355366, -247741639374, 211103573298, -131365867290,
        58189107627, -17410248271, 3158642445, -262747265
    ],
    [
        13064406523627, -61497552797274, 214696591002612, -524924579905150, 932884546055895, -1233589244941764,
        1226443086129408, -915883387152444, 507140369728425, -202322913738370, 55060974662412, -9160551085734,
        703604254357
    ],
    [
        27511554976875, -140970750679621, 537247052515662, -1445313351681906, 2854429571790805, -4246767353305755,
        4825671323488452, -4204551925534524, 2793869602879077, -1393306307155755, 505586141196430, -126174972681906,
        19382853593787, -1382741929621
    ],
    [
        173233498598849, -960122866404112, 3966421670215481, -11643637530577472, 25298910337081429, -41825269932507728,
        53471026659940509, -53246738660646912, 41280216336284259, -24704503655607728, 11205849753515179,
        -3728807256577472, 859236476684231, -122594813904112, 8164168737599
    ],
    [
        362555126427073, -2161567671248849, 9622096909515337, -30607373860520569, 72558117072259733,
        -131963191940828581, 187463140112902893, -210020588912321949, 186087544263596643, -129930094104237331,
        70724351582843483, -29417910911251819, 9038571752734087, -1934443196892599, 257650275915823, -16088129229375
    ],
    [
        192996103681340479, -1231887339593444974, 5878428128276811750, -20141834622844109630, 51733880057282977010,
        -102651404730855807942, 160414858999474733422, -199694296833704562550, 199061418623907202560,
        -158848144481581407370, 100878076849144434322, -50353311405771659322, 19338911944324897550,
        -5518639984393844930, 1102560345141059610, -137692773163513234, 8092989203533249
    ],
    [
        401972381695456831, -2735437642844079789, 13930159965811142228, -51150187791975812900, 141500575026572531760,
        -304188128232928718008, 518600355541383671092, -710171024091234303204, 786600875277595877750,
        -706174326992944287370, 512538584122114046748, -298477260353977522892, 137563142659866897224,
        -49070094880794267600, 13071639236569712860, -2448689255584545196, 287848942064256339, -15980174332775873
    ],
    [
        333374427829017307697, -2409687649238345289684, 13044139139831833251471, -51099831122607588046344,
        151474888613495715415020, -350702929608291455167896, 647758157491921902292692, -967713746544629658690408,
        1179078743786280451953222, -1176161829956768365219840, 960377035444205950813626, -639182123082298748001432,
        343690461612471516746028, -147118738993288163742312, 48988597853073465932820, -12236035290567356418552,
        2157574942881818312049, -239560589366324764716, 12600467236042756559
    ],
    [
        691668239157222107697, -5292843584961252933125, 30349492858024727686755, -126346544855927856134295,
        399537307669842150996468, -991168450545135070835076, 1971629028083798845750380, -3191065388846318679544380,
        4241614331208149947151790, -4654326468801478894406214, 4222756879776354065593786, -3161821089800186539248210,
        1943018818982002395655620, -970350191086531368649620, 387739787034699092364924, -121059601023985433003532,
        28462032496476316665705, -4740335757093710713245, 498669220956647866875, -24919383499187492303
    ],
]

_MOULTON_COEFFICIENTS = [
    [],  # order 0
    [1],
    [1, 1],
    [5, 8, -1],
    [9, 19, -5, 1],
    [251, 646, -264, 106, -19],
    [475, 1427, -798, 482, -173, 27],
    [19087, 65112, -46461, 37504, -20211, 6312, -863],
    [36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375],
    [1070017, 4467094, -4604594, 5595358, -5033120, 3146338, -1291214, 312874, -33953],
    [2082753, 9449717, -11271304, 16002320, -17283646, 13510082, -7394032, 2687864, -583435, 57281],
    [
        134211265, 656185652, -890175549, 1446205080, -1823311566, 1710774528, -1170597042, 567450984, -184776195,
        36284876, -3250433
    ],
    [
        262747265, 1374799219, -2092490673, 3828828885, -5519460582, 6043521486, -4963166514, 3007739418, -1305971115,
        384709327, -68928781, 5675265
    ],
    [
        703604254357, 3917551216986, -6616420957428, 13465774256510, -21847538039895, 27345870698436, -26204344465152,
        19058185652796, -10344711794985, 4063327863170, -1092096992268, 179842822566, -13695779093
    ],
    [
        1382741929621, 8153167962181, -15141235084110, 33928990133618, -61188680131285, 86180228689563, -94393338653892,
        80101021029180, -52177910882661, 25620259777835, -9181635605134, 2268078814386, -345457086395, 24466579093
    ],
    [
        8164168737599, 50770967534864, -102885148956217, 251724894607936, -499547203754837, 781911618071632,
        -963605400824733, 934600833490944, -710312834197347, 418551804601264, -187504936597931, 61759426692544,
        -14110480969927, 1998759236336, -132282840127
    ],
    [
        16088129229375, 105145058757073, -230992163723849, 612744541065337, -1326978663058069, 2285168598349733,
        -3129453071993581, 3414941728852893, -2966365730265699, 2039345879546643, -1096355235402331, 451403108933483,
        -137515713789319, 29219384284087, -3867689367599, 240208245823
    ],
    [
        8092989203533249, 55415287221275246, -131240807912923110, 375195469874202430, -880520318434977010,
        1654462865819232198, -2492570347928318318, 3022404969160106870, -2953729295811279360, 2320851086013919370,
        -1455690451266780818, 719242466216944698, -273894214307914510, 77597639915764930, -15407325991235610,
        1913813460537746, -111956703448001
    ],
    [
        15980174332775873, 114329243705491117, -290470969929371220, 890337710266029860, -2250854333681641520,
        4582441343348851896, -7532171919277411636, 10047287575124288740, -10910555637627652470, 9644799218032932490,
        -6913858539337636636, 3985516155854664396, -1821304040326216520, 645008976643217360, -170761422500096220,
        31816981024600492, -3722582669836627, 205804074290625
    ],
    [
        12600467236042756559, 93965550344204933076, -255007751875033918095, 834286388106402145800,
        -2260420115705863623660, 4956655592790542146968, -8827052559979384209108, 12845814402199484797800,
        -15345231910046032448070, 15072781455122686545920, -12155867625610599812538, 8008520809622324571288,
        -4269779992576330506540, 1814584564159445787240, -600505972582990474260, 149186846171741510136,
        -26182538841925312881, 2895045518506940460, -151711881512390095
    ],
    [
        24919383499187492303, 193280569173472261637, -558160720115629395555, 1941395668950986461335,
        -5612131802364455926260, 13187185898439270330756, -25293146116627869170796, 39878419226784442421820,
        -51970649453670274135470, 56154678684618739939910, -50320851025594566473146, 37297227252822858381906,
        -22726350407538133839300, 11268210124987992327060, -4474886658024166985340, 1389665263296211699212,
        -325187970422032795497, 53935307402575440285, -5652892248087175675, 281550972898020815
    ],
]

_DIVISOR = [
    None, 11, 2, 12, 24, 720, 1440, 60480, 120960, 3628800, 7257600, 479001600, 958003200, 2615348736000, 5230697472000,
    31384184832000, 62768369664000, 32011868528640000, 64023737057280000, 51090942171709440000, 102181884343418880000
]

_MIN_ORDER = 4
_MAX_ORDER = 12
_MAX_ITERS = 4


class AdamsBashforthMoulton(FixedGridODESolver):

    def __init__(
        self, func, y0, rtol=1e-3, atol=1e-4, implicit=True, max_iters=_MAX_ITERS, max_order=_MAX_ORDER, **kwargs
    ):
        super(AdamsBashforthMoulton, self).__init__(func, y0, **kwargs)

        self.rtol = rtol
        self.atol = atol
        self.implicit = implicit
        self.max_iters = max_iters
        self.max_order = int(min(max_order, _MAX_ORDER))
        self.prev_f = collections.deque(maxlen=self.max_order - 1)
        self.prev_t = None

    def _update_history(self, t, f):
        if self.prev_t is None or self.prev_t != t:
            self.prev_f.appendleft(f)
            self.prev_t = t

    def step_func(self, func, t, dt, y):
        self._update_history(t, func(t, y))
        order = min(len(self.prev_f), self.max_order - 1)
        if order < _MIN_ORDER - 1:
            # Compute using RK4.
            dy = rk4_alt_step_func(func, t, dt, y, k1=self.prev_f[0])
            return dy
        else:
            # Adams-Bashforth predictor.
            bashforth_coeffs = _BASHFORTH_COEFFICIENTS[order]
            ab_div = _DIVISOR[order]
            dy = tuple(dt * _scaled_dot_product(1 / ab_div, bashforth_coeffs, f_) for f_ in zip(*self.prev_f))

            # Adams-Moulton corrector.
            if self.implicit:
                moulton_coeffs = _MOULTON_COEFFICIENTS[order + 1]
                am_div = _DIVISOR[order + 1]
                delta = tuple(dt * _scaled_dot_product(1 / am_div, moulton_coeffs[1:], f_) for f_ in zip(*self.prev_f))
                converged = False
                for _ in range(self.max_iters):
                    dy_old = dy
                    f = func(t + dt, tuple(y_ + dy_ for y_, dy_ in zip(y, dy)))
                    dy = tuple(dt * (moulton_coeffs[0] / am_div) * f_ + delta_ for f_, delta_ in zip(f, delta))
                    converged = _has_converged(dy_old, dy, self.rtol, self.atol)
                    if converged:
                        break
                if not converged:
                    print('Warning: Functional iteration did not converge. Solution may be incorrect.', file=sys.stderr)
                    self.prev_f.pop()
                self._update_history(t, f)
            return dy

    @property
    def order(self):
        return 4


class AdamsBashforth(AdamsBashforthMoulton):

    def __init__(self, func, y0, **kwargs):
        super(AdamsBashforth, self).__init__(func, y0, implicit=False, **kwargs)
