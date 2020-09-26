import math
import torch
import signatory
from .misc import cheap_stack, validate_input_path, register_computed_parameter, tridiagonal_solve

__all__ = [
    'linear_interpolation_coeffs', 'natural_cubic_spline_coeffs',
    'LinearInterpolation', 'NaturalCubicSpline', 'logsignature_windows'
]

_two_pi = 2 * math.pi
_inv_two_pi = 1 / _two_pi


def _linear_interpolation_coeffs_with_missing_values_scalar(t, x):
    # t and X both have shape (length,)

    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        return torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

    if path_no_nan.size(0) == x.size(0):
        # Every entry is not-NaN, so just return.
        return x

    x = x.clone()
    # How to deal with missing values at the start or end of the time series? We impute an observation at the very start
    # equal to the first actual observation made, and impute an observation at the very end equal to the last actual
    # observation made, and then proceed as normal.
    if torch.isnan(x[0]):
        x[0] = path_no_nan[0]
    if torch.isnan(x[-1]):
        x[-1] = path_no_nan[-1]

    nan_indices = torch.arange(x.size(0), device=x.device).masked_select(torch.isnan(x))

    if nan_indices.size(0) == 0:
        # We only had missing values at the start or end
        return x

    prev_nan_index = nan_indices[0]
    prev_not_nan_index = prev_nan_index - 1
    prev_not_nan_indices = [prev_not_nan_index]
    for nan_index in nan_indices[1:]:
        if prev_nan_index != nan_index - 1:
            prev_not_nan_index = nan_index - 1
        prev_nan_index = nan_index
        prev_not_nan_indices.append(prev_not_nan_index)

    next_nan_index = nan_indices[-1]
    next_not_nan_index = next_nan_index + 1
    next_not_nan_indices = [next_not_nan_index]
    for nan_index in reversed(nan_indices[:-1]):
        if next_nan_index != nan_index + 1:
            next_not_nan_index = nan_index + 1
        next_nan_index = nan_index
        next_not_nan_indices.append(next_not_nan_index)
    next_not_nan_indices = reversed(next_not_nan_indices)
    for prev_not_nan_index, nan_index, next_not_nan_index in zip(prev_not_nan_indices,
                                                                 nan_indices,
                                                                 next_not_nan_indices):
        prev_stream = x[prev_not_nan_index]
        next_stream = x[next_not_nan_index]
        prev_time = t[prev_not_nan_index]
        next_time = t[next_not_nan_index]
        time = t[nan_index]
        ratio = (time - prev_time) / (next_time - prev_time)
        x[nan_index] = prev_stream + ratio * (next_stream - prev_stream)

    return x


def _linear_interpolation_coeffs_with_missing_values(t, x):
    if x.ndimension() == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _linear_interpolation_coeffs_with_missing_values_scalar(t, x)
    else:
        out_pieces = []
        for p in x.unbind(dim=0):  # TODO: parallelise over this
            out = _linear_interpolation_coeffs_with_missing_values(t, p)
            out_pieces.append(out)
        return cheap_stack(out_pieces, dim=0)


def linear_interpolation_coeffs(x, t=None):
    """Calculates the knots of the linear interpolation of the batch of controls given.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]).

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `torchcde.LinearInterpolation`.

        See the docstring for `torchcde.natural_cubic_spline_coeffs` for more information on why we do it this
        way.
    """
    t = validate_input_path(x, t)

    if torch.isnan(x).any():
        x = _linear_interpolation_coeffs_with_missing_values(t, x.transpose(-1, -2)).transpose(-1, -2)
    return x


class LinearInterpolation(torch.nn.Module):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, coeffs, t=None, reparameterise='none', **kwargs):
        """
        Arguments:
            coeffs: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed.)
            reparameterise: Either 'none' or 'bump'. Defaults to 'none'. Reparameterising each linear piece can help
                adaptive step size solvers, in particular those that aren't aware of where the kinks in the path are.
        """
        super(LinearInterpolation, self).__init__(**kwargs)
        assert reparameterise in ('none', 'bump')

        if t is None:
            t = torch.linspace(0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype, device=coeffs.device)

        derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

        register_computed_parameter(self, '_t', t)
        register_computed_parameter(self, '_coeffs', coeffs)
        register_computed_parameter(self, '_derivs', derivs)
        self._reparameterise = reparameterise

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._derivs.dtype, device=self._derivs.device)
        maxlen = self._derivs.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        prev_coeff = self._coeffs[..., index, :]
        next_coeff = self._coeffs[..., index + 1, :]
        prev_t = self._t[index]
        next_t = self._t[index + 1]
        diff_t = next_t - prev_t
        if self._reparameterise == 'bump':
            fractional_part = fractional_part - diff_t * _inv_two_pi * torch.sin(_two_pi * fractional_part / diff_t)
        return prev_coeff + fractional_part * (next_coeff - prev_coeff) / diff_t.unsqueeze(-1)

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        deriv = self._derivs[..., index, :]

        if self._reparameterise != 'none':
            prev_t = self._t[index]
            next_t = self._t[index + 1]
            diff_t = next_t - prev_t
            fractional_part = fractional_part / diff_t
            if self._reparameterise == 'bump':
                mult = 1 - torch.cos(_two_pi * fractional_part)
            else:
                raise RuntimeError

            deriv = deriv * mult
        return deriv


def _natural_cubic_spline_coeffs_without_missing_values(t, x):
    # x should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = x.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[..., 1:] - t[..., :1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    else:
        # Set up some intermediate values
        time_diffs = t[1:] - t[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=x.dtype, device=x.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
                                                  time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


def _natural_cubic_spline_coeffs_with_missing_values(t, x):
    if x.ndimension() == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x)
    else:
        a_pieces = []
        b_pieces = []
        two_c_pieces = []
        three_d_pieces = []
        for p in x.unbind(dim=0):  # TODO: parallelise over this
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p)
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (cheap_stack(a_pieces, dim=0),
                cheap_stack(b_pieces, dim=0),
                cheap_stack(two_c_pieces, dim=0),
                cheap_stack(three_d_pieces, dim=0))


def _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x):
    # t and x both have shape (length,)

    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        # Note that we may assume that X.size(0) >= 2 by the checks in __init__ so "X.size(0) - 1" is a valid
        # thing to do.
        return (torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device))
    # else we have at least one non-NaN entry, in which case we're going to impute at least one more entry (as
    # the path is of length at least 2 so the start and the end aren't the same), so we will then have at least two
    # non-Nan entries. In particular we can call _compute_coeffs safely later.

    # How to deal with missing values at the start or end of the time series? We're creating some splines, so one
    # option is just to extend the first piece backwards, and the final piece forwards. But polynomials tend to
    # behave badly when extended beyond the interval they were constructed on, so the results can easily end up
    # being awful.
    # Instead we impute an observation at the very start equal to the first actual observation made, and impute an
    # observation at the very end equal to the last actual observation made, and then proceed with splines as
    # normal.
    need_new_not_nan = False
    if torch.isnan(x[0]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[0] = path_no_nan[0]
    if torch.isnan(x[-1]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[-1] = path_no_nan[-1]
    if need_new_not_nan:
        not_nan = ~torch.isnan(x)
        path_no_nan = x.masked_select(not_nan)
    times_no_nan = t.masked_select(not_nan)

    # Find the coefficients on the pieces we do understand
    # These all have shape (len - 1,)
    (a_pieces_no_nan,
     b_pieces_no_nan,
     two_c_pieces_no_nan,
     three_d_pieces_no_nan) = _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)

    # Now we're going to normalise them to give coefficients on every interval
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []

    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan))
    next_time_no_nan = next(iter_times_no_nan)
    for time in t[:-1]:
        # will always trigger on the first iteration because of how we've imputed missing values at the start and
        # end of the time series.
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0))


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def natural_cubic_spline_coeffs(x, t=None):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]).

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `torchcde.NaturalCubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `torchcde.NaturalCubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call NaturalCubicSpline as the first part of your model.

        See also the accompanying example.py.
    """
    t = validate_input_path(x, t)

    if torch.isnan(x).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, x.transpose(-1, -2))
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, x.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    two_c = two_c.transpose(-1, -2)
    three_d = three_d.transpose(-1, -2)
    coeffs = torch.cat([a, b, two_c, three_d], dim=-1)  # for simplicity put them all together
    return coeffs


class NaturalCubicSpline(torch.nn.Module):
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(x)
        # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(coeffs)
        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by `torchcde.natural_cubic_spline_coeffs`.
            t: As passed to linear_interpolation_coeffs. (If it was passed.)
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        if t is None:
            t = torch.linspace(0, coeffs.size(-2), coeffs.size(-2) + 1, dtype=coeffs.dtype, device=coeffs.device)

        channels = coeffs.size(-1) // 4
        if channels * 4 != coeffs.size(-1):  # check that it's a multiple of 4
            raise ValueError("Passed invalid coeffs.")
        a, b, two_c, three_d = (coeffs[..., :channels], coeffs[..., channels:2 * channels],
                                coeffs[..., 2 * channels:3 * channels], coeffs[..., 3 * channels:])

        register_computed_parameter(self, '_t', t)
        register_computed_parameter(self, '_a', a)
        register_computed_parameter(self, '_b', b)
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        register_computed_parameter(self, '_two_c', two_c)
        register_computed_parameter(self, '_three_d', three_d)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._b.dtype,  device=self._b.device)
        maxlen = self._b.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv


def logsignature_windows(x, depth, window_length, t=None):
    """Calculates logsignatures over multiple windows, for the batch of controls given, as in the log-ODE method.

    This corresponds to a transform of the time series, and should be used prior to applying one of the interpolation
    schemes.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        depth: What depth to compute the logsignatures to.
        window_length: How long a time interval to compute logsignatures over.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]).

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tuple of two tensors, which are the values and times of the transformed path.
    """
    t = validate_input_path(x, t)

    # slightly roundabout way of doing things (rather than using arange) so that it's constructed differentiably
    timespan = t[-1] - t[0]
    num_pieces = (timespan / window_length).ceil().to(int).item()
    end_t = t[0] + num_pieces * window_length
    new_t = torch.linspace(t[0], end_t, num_pieces + 1, dtype=t.dtype, device=t.device)
    new_t = torch.min(new_t, t.max())

    t_index = 0
    new_t_unique = []
    new_t_indices = []
    for new_t_elem in new_t:
        while True:
            lequal = (new_t_elem <= t[t_index])
            close = new_t_elem.allclose(t[t_index])
            if lequal or close:
                break
            t_index += 1
        new_t_indices.append(t_index + len(new_t_unique))
        if close:
            continue
        new_t_unique.append(new_t_elem.unsqueeze(0))

    batch_dimensions = x.shape[:-2]

    missing_X = torch.full((1,), float('nan'), dtype=x.dtype, device=x.device).expand(*batch_dimensions, 1, x.size(-1))
    if len(new_t_unique) > 0:  # no-op if len == 0, so skip for efficiency
        t, indices = torch.cat([t, *new_t_unique]).sort()
        x = torch.cat([x, missing_X], dim=-2)[..., indices.clamp(0, x.size(-2)), :]

    # Fill in any missing data linearly (linearly because that's what signatures do in between observations anyway)
    # and conveniently that's what this already does. Here 'missing data' includes the NaNs we've just added.
    x = linear_interpolation_coeffs(x, t)

    # Flatten batch dimensions for compatibility with Signatory
    flatten_X = x.view(-1, x.size(-2), x.size(-1))
    first_increment = torch.zeros(*batch_dimensions, signatory.logsignature_channels(x.size(-1), depth), dtype=x.dtype,
                                  device=x.device)
    first_increment[..., :x.size(-1)] = x[..., 0, :]
    logsignatures = [first_increment]
    compute_logsignature = signatory.Logsignature(depth=depth)
    for index, next_index, time, next_time in zip(new_t_indices[:-1], new_t_indices[1:], new_t[:-1], new_t[1:]):
        logsignature = compute_logsignature(flatten_X[..., index:next_index + 1, :])
        logsignature = logsignature.view(*batch_dimensions, -1) * (next_time - time)
        logsignatures.append(logsignature)

    logsignatures = torch.stack(logsignatures, dim=-2)
    logsignatures = logsignatures.cumsum(dim=-2)

    return logsignatures, new_t