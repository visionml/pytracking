import torch
from pytracking.libs.tensorlist import tensor_operation


def is_complex(a: torch.Tensor) -> bool:
    return a.dim() >= 4 and a.shape[-1] == 2


def is_real(a: torch.Tensor) -> bool:
    return not is_complex(a)


@tensor_operation
def mult(a: torch.Tensor, b: torch.Tensor):
    """Pointwise complex multiplication of complex tensors."""

    if is_real(a):
        if a.dim() >= b.dim():
            raise ValueError('Incorrect dimensions.')
        # a is real
        return mult_real_cplx(a, b)
    if is_real(b):
        if b.dim() >= a.dim():
            raise ValueError('Incorrect dimensions.')
        # b is real
        return mult_real_cplx(b, a)

    # Both complex
    c = mult_real_cplx(a[..., 0], b)
    c[..., 0] -= a[..., 1] * b[..., 1]
    c[..., 1] += a[..., 1] * b[..., 0]
    return c


@tensor_operation
def mult_conj(a: torch.Tensor, b: torch.Tensor):
    """Pointwise complex multiplication of complex tensors, with conjugate on b: a*conj(b)."""

    if is_real(a):
        if a.dim() >= b.dim():
            raise ValueError('Incorrect dimensions.')
        # a is real
        return mult_real_cplx(a, conj(b))
    if is_real(b):
        if b.dim() >= a.dim():
            raise ValueError('Incorrect dimensions.')
        # b is real
        return mult_real_cplx(b, a)

    # Both complex
    c = mult_real_cplx(b[...,0], a)
    c[..., 0] += a[..., 1] * b[..., 1]
    c[..., 1] -= a[..., 0] * b[..., 1]
    return c


@tensor_operation
def mult_real_cplx(a: torch.Tensor, b: torch.Tensor):
    """Pointwise complex multiplication of real tensor a with complex tensor b."""

    if is_real(b):
        raise ValueError('Last dimension must have length 2.')

    return a.unsqueeze(-1) * b


@tensor_operation
def div(a: torch.Tensor, b: torch.Tensor):
    """Pointwise complex division of complex tensors."""

    if is_real(b):
        if b.dim() >= a.dim():
            raise ValueError('Incorrect dimensions.')
        # b is real
        return div_cplx_real(a, b)

    return div_cplx_real(mult_conj(a, b), abs_sqr(b))


@tensor_operation
def div_cplx_real(a: torch.Tensor, b: torch.Tensor):
    """Pointwise complex division of complex tensor a with real tensor b."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a / b.unsqueeze(-1)


@tensor_operation
def abs_sqr(a: torch.Tensor):
    """Squared absolute value."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return torch.sum(a*a, -1)


@tensor_operation
def abs(a: torch.Tensor):
    """Absolute value."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return torch.sqrt(abs_sqr(a))


@tensor_operation
def conj(a: torch.Tensor):
    """Complex conjugate."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    # return a * torch.Tensor([1, -1], device=a.device)
    return complex(a[...,0], -a[...,1])


@tensor_operation
def real(a: torch.Tensor):
    """Real part."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a[..., 0]


@tensor_operation
def imag(a: torch.Tensor):
    """Imaginary part."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a[..., 1]


@tensor_operation
def complex(a: torch.Tensor, b: torch.Tensor = None):
    """Create complex tensor from real and imaginary part."""

    if b is None:
        b = a.new_zeros(a.shape)
    elif a is None:
        a = b.new_zeros(b.shape)

    return torch.cat((a.unsqueeze(-1), b.unsqueeze(-1)), -1)


@tensor_operation
def mtimes(a: torch.Tensor, b: torch.Tensor, conj_a=False, conj_b=False):
    """Complex matrix multiplication of complex tensors.
    The dimensions (-3, -2) are matrix multiplied. -1 is the complex dimension."""

    if is_real(a):
        if a.dim() >= b.dim():
            raise ValueError('Incorrect dimensions.')
        return mtimes_real_complex(a, b, conj_b=conj_b)
    if is_real(b):
        if b.dim() >= a.dim():
            raise ValueError('Incorrect dimensions.')
        return mtimes_complex_real(a, b, conj_a=conj_a)

    if not conj_a and not conj_b:
        return complex(torch.matmul(a[..., 0], b[..., 0]) - torch.matmul(a[..., 1], b[..., 1]),
                       torch.matmul(a[..., 0], b[..., 1]) + torch.matmul(a[..., 1], b[..., 0]))
    if conj_a and not conj_b:
        return complex(torch.matmul(a[..., 0], b[..., 0]) + torch.matmul(a[..., 1], b[..., 1]),
                       torch.matmul(a[..., 0], b[..., 1]) - torch.matmul(a[..., 1], b[..., 0]))
    if not conj_a and conj_b:
        return complex(torch.matmul(a[..., 0], b[..., 0]) + torch.matmul(a[..., 1], b[..., 1]),
                       torch.matmul(a[..., 1], b[..., 0]) - torch.matmul(a[..., 0], b[..., 1]))
    if conj_a and conj_b:
        return complex(torch.matmul(a[..., 0], b[..., 0]) - torch.matmul(a[..., 1], b[..., 1]),
                       -torch.matmul(a[..., 0], b[..., 1]) - torch.matmul(a[..., 1], b[..., 0]))


@tensor_operation
def mtimes_real_complex(a: torch.Tensor, b: torch.Tensor, conj_b=False):
    if is_real(b):
        raise ValueError('Incorrect dimensions.')

    if not conj_b:
        return complex(torch.matmul(a, b[..., 0]), torch.matmul(a, b[..., 1]))
    if conj_b:
        return complex(torch.matmul(a, b[..., 0]), -torch.matmul(a, b[..., 1]))


@tensor_operation
def mtimes_complex_real(a: torch.Tensor, b: torch.Tensor, conj_a=False):
    if is_real(a):
        raise ValueError('Incorrect dimensions.')

    if not conj_a:
        return complex(torch.matmul(a[..., 0], b), torch.matmul(a[..., 1], b))
    if conj_a:
        return complex(torch.matmul(a[..., 0], b), -torch.matmul(a[..., 1], b))


@tensor_operation
def exp_imag(a: torch.Tensor):
    """Complex exponential with imaginary input: e^(i*a)"""

    a = a.unsqueeze(-1)
    return torch.cat((torch.cos(a), torch.sin(a)), -1)



