import scipy.special
import torch
import numpy as np


class Ive(torch.autograd.Function):
    """
    The modified Bessel of the first kind, I_v(x), exponentially scaled by
    e^(-x) and made differentiable with respect to the continuous variable, x.
    """

    @staticmethod
    def forward(ctx, v: float, x: float):
        ctx._v = v
        ctx.save_for_backward(x)

        x_cpu = x.detach().cpu().numpy()

        # TODO: Original paper uses `np.close(v, 0)` and `np.close(v, 1)` to
        # default to scipy.special.i0e and scipy.special.i0e. Is this needed?
        output = scipy.special.ive(
            v, x_cpu, dtype=x_cpu.dtype
        )  # pylint: disable=no-member

        return torch.Tensor(output).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        v = ctx._v
        (x,) = ctx.saved_tensors

        # Only compute a gradient for x, return None for order, v.
        return (
            None,
            grad_output * (Ive.apply(v - 1, x) + Ive.apply(v, x) * ((v + x) / x)),
        )


ive = Ive.apply
