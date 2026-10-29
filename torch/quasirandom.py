# mypy: allow-untyped-defs
# [MODULE DOC] Defines SobolEngine, a generator of (optionally scrambled) Sobol low-discrepancy
# quasi-random sequences backed by C++ ops, supporting up to 21201 dimensions. No import-time
# side effects.

import torch


class SobolEngine:
    # [CLASS DOC] Stateful generator of Sobol quasi-random points in the unit hypercube.
    # State: dimension, scramble flag, seed, the direction-number matrix (sobolstate),
    # a shift vector (zero, or random when scrambled), the running integer point (quasi),
    # the cached first point, and the count of points generated so far. Invariants:
    # 1 <= dimension <= MAXDIM; points are produced sequentially so draws depend on history.
    # Key methods: draw, draw_base2, reset, fast_forward.
    r"""
    The :class:`torch.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``

    Examples::

        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    """

    MAXBIT = 30
    MAXDIM = 21201

    def __init__(self, dimension, scramble=False, seed=None):
        # [FUNCTION DOC] Validates dimension (raises ValueError outside 1..MAXDIM), stores
        # settings, builds the CPU direction-number state via a C++ op, then sets the shift:
        # zeros, or a random scramble (which also scrambles the state) when scramble is true.
        # Initializes the running point from the shift, caches the first point scaled to
        # [0,1), and sets the generated count to zero.
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        cpu = torch.device("cpu")

        self.sobolstate = torch.zeros(
            dimension, self.MAXBIT, device=cpu, dtype=torch.long
        )
        torch._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        if not self.scramble:
            self.shift = torch.zeros(self.dimension, device=cpu, dtype=torch.long)
        else:
            self._scramble()

        self.quasi = self.shift.clone(memory_format=torch.contiguous_format)
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        self.num_generated = 0

    def draw(
        self,
        n: int = 1,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Returns the next n points as an (n, dimension) tensor of the given
        # dtype (default: current default dtype), advancing the engine state and count.
        # The very first point comes from the cached shift point; later points come from a
        # C++ draw op. If out is given, it is resized, filled with the result and returned.
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        if self.num_generated == 0:
            if n == 1:
                result = self._first_point.to(dtype)
            else:
                result, self.quasi = torch._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                result = torch.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            result, self.quasi = torch._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        self.num_generated += n

        if out is not None:
            out.resize_as_(result).copy_(result)
            return out

        return result

    def draw_base2(
        self,
        m: int,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Draws 2**m points via draw. Raises ValueError unless the total
        # count after drawing (already generated plus 2**m) is a power of two, since only
        # then do Sobol balance properties hold. Same out/dtype semantics as draw.
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        n = 2**m
        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        # [FUNCTION DOC] Restores the running point to the shift and the generated count to
        # zero, so drawing restarts the sequence. Returns the engine itself.
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        # [FUNCTION DOC] Advances the state by n points without producing them, using a C++
        # in-place op (offset by one on the first call, since the first point is the shift),
        # and increases the generated count by n. Returns the engine; n of zero is a no-op.
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        if n == 0:
            return self
        if self.num_generated == 0:
            torch._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            torch._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        self.num_generated += n
        return self

    def _scramble(self):
        # [FUNCTION DOC] HELPER: Draws random bits (from a CPU generator seeded with the
        # engine seed if given, else the global RNG) to form the per-dimension integer shift
        # vector and random lower-triangular binary matrices, then applies the matrices to
        # the direction-number state in place via a C++ op. Sets the shift attribute.
        g: torch.Generator | None = None
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)

        cpu = torch.device("cpu")

        # Generate shift vector
        shift_ints = torch.randint(
            2, (self.dimension, self.MAXBIT), device=cpu, generator=g
        )
        self.shift = torch.mv(
            shift_ints, torch.pow(2, torch.arange(0, self.MAXBIT, device=cpu))
        )

        # Generate lower triangular matrices (stacked across dimensions)
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = torch.randint(2, ltm_dims, device=cpu, generator=g).tril()

        torch._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)

    def __repr__(self):
        # [FUNCTION DOC] Returns a string like the class name with dimension, plus scramble
        # and seed entries when they are set.
        fmt_string = [f"dimension={self.dimension}"]
        if self.scramble:
            fmt_string += ["scramble=True"]
        if self.seed is not None:
            fmt_string += [f"seed={self.seed}"]
        return self.__class__.__name__ + "(" + ", ".join(fmt_string) + ")"
