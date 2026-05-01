"""
harness/counters.py — Lightweight eval-counting wrapper.
"""


class EvalCounter:
    """Wraps (f, grad, hess) callables, exposing them as .f / .grad / .hess."""

    def __init__(self, f, grad, hess):
        self._f    = f
        self._grad = grad
        self._hess = hess
        self.n_f    = 0
        self.n_grad = 0
        self.n_hess = 0

    def f(self, x):
        self.n_f += 1
        return self._f(x)

    def grad(self, x):
        self.n_grad += 1
        return self._grad(x)

    def hess(self, x):
        self.n_hess += 1
        return self._hess(x)
