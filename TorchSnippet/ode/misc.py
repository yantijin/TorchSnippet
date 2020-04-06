import torch
import warnings


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: unexpected arguments {}'.format(
            solver.__class__.__name__, unused_kwargs
        ))


def _assert_increasing(t):
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing'


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _norm(x):
    """Compute RMS norm."""
    if torch.is_tensor(x):
        return x.norm() / (x.numel()**0.5)
    else:
        return torch.sqrt(sum(x_.norm()**2 for x_ in x) / sum(x_.numel() for x_ in x))


def _check_inputs(func, y0, t):
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)
        _base_nontuple_func_ = func
        func = lambda t, y: (_base_nontuple_func_(t, y[0]),)
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    if _decreasing(t):
        t = -t
        _base_reverse_func = func
        func = lambda t, y: tuple(-f_ for f_ in _base_reverse_func(-t, y))

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])


def _convert_to_tensor(a, dtype=None, device=None):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if dtype is not None:
        a = a.type(dtype)
    if device is not None:
        a = a.to(device)
    return a


def _compute_error_ratio(error_estimate, error_tol=None, rtol=None, atol=None, y0=None, y1=None):
    if error_tol is None:
        assert rtol is not None and atol is not None and y0 is not None and y1 is not None
        rtol if _is_iterable(rtol) else [rtol] * len(y0)
        atol if _is_iterable(atol) else [atol] * len(y0)
        error_tol = tuple(
            atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(y1_))
            for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1)
        )
    error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_, error_tol_ in zip(error_estimate, error_tol))
    mean_sq_error_ratio = tuple(torch.mean(error_ratio_ * error_ratio_) for error_ratio_ in error_ratio)
    return mean_sq_error_ratio


def _optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next step."""
    mean_error_ratio = max(mean_error_ratio)  # Compute step size based on highest ratio.
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=torch.float64, device=mean_error_ratio.device)
    error_ratio = torch.sqrt(mean_error_ratio).to(last_step)
    exponent = torch.tensor(1 / order).to(last_step)
    factor = torch.max(1 / ifactor, torch.min(error_ratio**exponent / safety, 1 / dfactor))
    return last_step / factor


def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([x * y for x, y in zip(xs, ys)])


def _is_iterable(inputs):
    try:
        iter(inputs)
        return True
    except TypeError:
        return False


def _select_initial_step(fun, t0, y0, order, rtol, atol, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    t0 = t0.to(y0[0])
    if f0 is None:
        f0 = fun(t0, y0)

    rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
    atol = atol if _is_iterable(atol) else [atol] * len(y0)

    scale = tuple(atol_ + torch.abs(y0_) * rtol_ for y0_, atol_, rtol_ in zip(y0, atol, rtol))

    d0 = tuple(_norm(y0_ / scale_) for y0_, scale_ in zip(y0, scale))
    d1 = tuple(_norm(f0_ / scale_) for f0_, scale_ in zip(f0, scale))

    if max(d0).item() < 1e-5 or max(d1).item() < 1e-5:
        h0 = torch.tensor(1e-6).to(t0)
    else:
        h0 = 0.01 * max(d0_ / d1_ for d0_, d1_ in zip(d0, d1))

    y1 = tuple(y0_ + h0 * f0_ for y0_, f0_ in zip(y0, f0))
    f1 = fun(t0 + h0, y1)

    d2 = tuple(_norm((f1_ - f0_) / scale_) / h0 for f1_, f0_, scale_ in zip(f1, f0, scale))

    if max(d1).item() <= 1e-15 and max(d2).item() <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6).to(h0), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1 + d2))**(1. / float(order + 1))

    return torch.min(100 * h0, h1)


def _has_converged(y0, y1, rtol, atol):
    """Checks that each element is within the error tolerance."""
    error_tol = tuple(atol + rtol * torch.max(torch.abs(y0_), torch.abs(y1_)) for y0_, y1_ in zip(y0, y1))
    error = tuple(torch.abs(y0_ - y1_) for y0_, y1_ in zip(y0, y1))
    return all((error_ < error_tol_).all() for error_, error_tol_ in zip(error, error_tol))


def _is_finite(tensor):
    _check = (tensor == float('inf')) + (tensor == float('-inf')) + torch.isnan(tensor)
    return not _check.any()


def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = tuple(
        _dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    b = tuple(
        _dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    c = tuple(
        _dot_product([-4 * dt, dt, -11, -5, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    d = tuple(dt * f0_ for f0_ in f0)
    e = y0
    return [a, b, c, d, e]


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    dtype = coefficients[0][0].dtype
    device = coefficients[0][0].device

    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    t1 = _convert_to_tensor(t1, dtype=dtype, device=device)
    t = _convert_to_tensor(t, dtype=dtype, device=device)

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = ((t - t0) / (t1 - t0)).type(dtype).to(device)

    xs = [torch.tensor(1).type(dtype).to(device), x]
    for _ in range(2, len(coefficients)):
        xs.append(xs[-1] * x)

    return tuple(_dot_product(coefficients_, reversed(xs)) for coefficients_ in zip(*coefficients))
