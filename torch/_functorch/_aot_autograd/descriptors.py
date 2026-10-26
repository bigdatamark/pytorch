# [MODULE DOC] Defines the AOTInput / AOTOutput descriptor hierarchy: frozen
# dataclasses that describe the semantic origin of each input and the semantic
# destination of each output of the FX graphs AOTAutograd produces (plain args,
# params, buffers, tangents, grads, subclass pieces, RNG state, effect tokens,
# view/synthetic bases, mutations, saved-for-backward values).
# Each descriptor exposes expr() (a human-readable path string) and some expose
# classification predicates (is_param, is_buffer, is_tangent, is_grad).
# No import-time side effects beyond class definitions; only dataclasses is imported.
"""
AOTAutograd descriptors are a path-like data structure (similar to pytree
paths and sources) that describe the semantic meaning of an input/output to FX
graphs.  Although you may know the input/output meaning at the top level of
the original function you traced, because we have many graph capture wrappers
that change the calling convention, it can be difficult to tell how these
correspond to the actual FX graph you get back, to say nothing about the extra
arguments/outputs for tangents, gradients, etc.  Descriptors describe the meaning
of arguments.

Examples
--------

Before we talk about the precise semantics, it's helpful to look at some
examples to get some intuition for the meaning of descriptors.  Here are some
input descriptors you might find on the joint FX graph:

* PlainAOTInput(idx=0) - the first input from the original callable, as is

* ParamAOTInput(target="mod.weight") - the parameter with FQN mod.weight

* TangentAOTInput(output=PlainAOTOutput(idx=1)) - the input tangent
  corresponding to the gradients for the second output in the forward graph

* ViewBaseAOTInput(base_of=PlainAOTInput(idx=0)) - it turned out the first
  input was actually a (differentiable) view of a tensor which aliased with
  another input tensor.  We replaced this input with a single input for the
  base of all of these inputs, replacing the original inputs (one of which is
  mentioned in base_of).  We would generate a GradAOTOutput for *this* input
  (and not the original PlainAOTInputs!)  If you have a joint graph where a
  view base like this is undesirable, you can eliminate this by cloning
  the views outside of the compiled region (assuming you aren't mutating this
  tensor).

* SubclassGetAttrAOTInput(base=AOTInput(idx=0), attr="inner") - this tensor
  corresponds to the "inner" tensor from the tensor subclass that is at the
  first index.  In general, joint graphs from AOTAutograd never take tensor
  subclasses as inputs; they are always unpacked into their constituent plain
  tensor pieces; use the descriptors to identify the parts of the tensor that
  are related.  Note that this can be nested (if you have nested tensor
  subclasses!)

Here are some output descriptors you might find on the Joint FX graph:

* PlainAOTOutput(idx=0) - the first output from the original forward function,
  as is

* GradAOTOutput(grad_of=PlainAOTInput(idx=1)) - the computed gradient for the
  second input to the graph, an output of the backward graph

* InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=0)) - when the first
  input is mutated, the new value to be copied into the first input of the
  graph.  Sometimes, these outputs can be elided and the ``copy_`` is done directly
  in the graph (controlled by keep_input_mutations), but if the input
  mutation must be differentiated through we always generate an output like this

* IntermediateBaseAOTOutput(base_of=PlainAOTOutput(idx=0)) - if we return
  multiple outputs which alias each other, we instead replace them with a single
  output tensor representing the base of all the aliases.  This output indicates
  it is the base for /one/ of those original outputs.  If this is undesirable in
  the joint graph, clone all outputs before returning from the graph.

* SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), idx="inner") - this
  tensor corresponds to the inner tensor of the first original output which
  is a tensor subclass.  This and other subclass components of that output will
  get repacked into a tensor subclass.

High level semantics
--------------------

OK, let's formally define a descriptor.  Intuitively, suppose we have::

    def wrapped_graph(*args):
        ret = graph(*in_transform(args))
        return out_transform(ret)

Then the descriptor for input[i] to graph describes a function fin_i such that::

    fin_i(args) == in_transform(args)[i]

and the descriptor for output[j] from graph describes a function fout_j such that::

    fout_j(out_transform(ret)) == ret[j]

AKA input descriptors tell you how to get from outer inputs to inner inputs,
while output descriptors tell you how to get from outer outputs to inner
outputs (inverse data flow!)

We haven't said anything about what these transformations actually do.  There
are three major transformations AOTAutograd does (performed in this order):

* View/mutation handling
* Autograd
* Subclasses

So intuitively, descriptors are built like this:

1. **PlainAOTInput, PlainAOTOutput.**

   We start off descriptors describing the exact inputs/outputs of the
   original flattened user function.  This user function is assumed to already
   be flattened; you would chain on pytree KeyPaths to further describe where
   in the pytree each input/output lived if you needed to deal with
   unflattened functions: this can be done from userland on top of
   descriptors, so the main descriptors mechanism doesn't handle it.

2. **SyntheticBaseAOTInput, ViewBaseAOTInput, MetadataMutationAOTOutput,
   InputMutationAOTOutput, IntermediateBaseAOTOutput**

   We deal with mutations and aliasing by removing duplicate PlainAOTInputs
   and introduce some new artificial inputs/outputs.  These inputs do not
   have a straightforward correspondence to the original user inputs, but if
   you are implementing a pass that doesn't care about the exact semantics of
   inputs, you should handle all of these uniformly in the same way as regular
   inputs.

3. **TangentAOTInput, GradAOTOutput**

   We deal with autograd by introducing a tangent input for every
   differentiable AOTOutput (including the new ones introduced above), and a
   gradient output for every differentiable AOTInput (also including new ones
   introduced above.) The arguments to these AOTInput/AOTOutput can ONLY be
   the ones we already have above (from steps 1-2).  As AOTAutograd does not
   currently support double backwards, you never have tangents of grads or
   vice versa (but in the future we could!)

4. **SubclassGetAttrAOTInput, SubclassGetAttrAOTOutput, et al.**

   We deal with subclasses by introducing flattened inputs/outputs (including
   potentially symbolic sizes/strides) for every AOTInput/AOTOutput that was a
   subclass.  As above, the arguments to these AOTInput/AOTOutput can ONLY be
   the ones we have above (from steps 1-3).  Recursive subclasses are
   supported, so these descriptors can nest with each other (so descriptors
   from step 4 are fair game as well.)

5. **ForwardTokenAOTInput, ForwardTokenAOTOutput, BackwardTokenAOTInput, BackwardTokenAOTOutput.**

   Some extra token inputs/outputs get added, these are synthetic and are just here to
   prevent DCE/reordering.

The important thing about the pipeline is that descriptors can ONLY be
created from top-to-bottom.  So for example, you can have::

    SubclassGetAttrAOTInput(TangentAOTInput(PlainAOTOutput(...)))  # OK

As you can see that PlainAOTOutput -> TangentAOTInput ->
SubclassGetAttrAOTInput is consistent with the pipeline ordering), but you can
NEVER have::

    TangentAOTInput(SubclassGetAttrAOTOutput(PlainAOTOutput(...))  # BAD

This is inconsistent; we always do autograd BEFORE we process subclasses!

Similarly, for example, this is illegal::

    GradAOTOutput(SubclassGetAttrAOTInput(PlainAOTInput(...)))  # BAD

It is illegal because subclasses are handled *after* create joint during
wrapper construction.  Instead, you would have::

    SubclassGetAttrAOTOutput(GradAOTOutput(PlainAOTInput(...)))  # OK

This intuitively captures the fact that we always to autograd directly on the
subclass, rather than after desugaring the subclass into its inner tensors.

Descriptor index
----------------

Here is a list of all AOTInput/AOTOutput, organized by how likely you need to
handle them:

* AOTInput

  * Important:

    * PlainAOTInput (the primals!)
    * ParamAOTInput
    * TangentAOTInput
    * SubclassGetAttrAOTInput et al. (if you use subclasses)

  * View related (can be eliminated by cloning inputs to graph; if you don't
    eliminate them, make sure to handle pairing them with GradAOTOutput):

    * ViewBaseAOTInput
    * SyntheticBaseAOTInput

  * Non-tensor, mostly just ignore them:

    * DummyAOTInput
    * PhiloxForwardSeedAOTInput
    * PhiloxForwardBaseOffsetAOTInput
    * PhiloxBackwardSeedAOTInput
    * PhiloxBackwardBaseOffsetAOTInput
    * ForwardTokenAOTInput
    * BackwardTokenAOTInput

* AOTOutput

  * Important:

    * PlainAOTOutput
    * GradAOTOutput
    * SubclassGetAttrAOTOutput et al. (if you use subclasses)

  * More obscure (if not eliminated, make sure you handle pairing them with
    TangentAOTInput):

    * InputMutationAOTOutput (can be eliminated if mutations are non-differentiable)
    * IntermediateBaseAOTOutput (can be eliminated by cloning outputs of graph)
    * MetadataMutationAOTOutput (uhh, just don't mutate metadata?)

  * Non-tensor, mostly just ignore them:

    * PhiloxUpdatedForwardOffsetAOTOutput
    * PhiloxUpdatedBackwardOffsetAOTOutput
    * ForwardTokenAOTOutput
    * BackwardTokenAOTOutput
    * DummyAOTOutput

For convenience, we also have DifferentiableAOTInput and
DifferentiableAOTOutput to help you classify which inputs/outputs can be
wrapped by GradAOTOutput/TangentAOTInput (respectively), which are essentially
all tensor AOTInput/AOTOutput excluding the subclass descriptors.

Implementation details
----------------------

The stylized view above is good for understanding how to interpret
descriptors, but the way that descriptors are generated in code is a bit more
complicated.  Specifically, AOTAutograd is structured as a series of wrappers
on the original user function, which are composed together to form the final
function to trace.  As a result of this, AOTAutograd ends up first building
the full AOTInputs for a function to be traced (as it builds the wrappers and
modifies the flat arguments to be compatible with the new input signature of
the wrapper), and then in reverse builds up the AOTOutput as it is tracing.

There is one major exception to this general idea of "build AOTInput first",
and then "build AOTOutput second": when we create TangentAOTInput, we need to
reference AOTOutputs (which output we are the tangents of) which we generally
haven't created yet.  There's two ways we deal with this:

- After the precompile steps (dedup and synthetic base handling), we do an
  initial pass to collect forward metadata that produces the initial set of
  PlainAOTOutputs which we use to create the tangent inputs.

- We also sometimes just violate causality and predict that an AOTOutput will
  be created in a particular way at some later point in time when we build an
  AOTInput.

As of July 2025, here is an exhaustive description of how inputs/outputs
traverse the wrappers from AOTAutograd, and what descriptors can be introduced
at these phases.

::

                                Build wrappers (FLOWS DOWN)         Run trace (FLOWS UP)
    -------------------------------------------------------------------------------------------------
    Begin                       PlainAOTInput                       (n/a)
                                ParamAOTInput

    Precompile dedupe           (remove dupes)                      (nothing)

    Precompile synthetic base   SyntheticBaseAOTInput               MetadataMutationAOTOutput
                                ViewBaseAOTInput

    Forward metadata trace      PlainAOTOutput                      (n/a)
                                MetadataMutationAOTOutput

    Prepare for autograd        (nothing)                           InputMutationAOTOutput
                                                                    IntermediateBaseAOTOutput

    Create joint                TangentAOTInput                     GradAOTOutput
                                w/ InputMutationAOTOutput
                                w/ IntermediateBaseAOTOutput

    Precompile subclass         SubclassGetAttrAOTInput et al.      SubclassGetAttrAOTOutput et al.

    Effect tokens               ForwardTokenAOTInput                ForwardTokenAOTOutput
                                BackwardTokenAOTInput               BackwardTokenAOTOutput

    End                         (n/a)                               PlainAOTOutput

It can be helpful to separately write down the input flow and the output flow
for ease of understanding the data flow:

* Input desc propagation (happens as we build wrappers)

  * [IN] Begin with original calling convention (PlainAOTInput, ParamAOTInput)
  * [IN] Precompile dedupe: (removes duplicate AOTInputs)
  * [IN] Precompile synthetic base: SyntheticBaseAOTInput, ViewBaseAOTInput
  * Forward metadata trace (mini output desc propagation)

    * [OUT] Original output convention: PlainAOTOutput
    * [OUT] Precompile synthetic base: MetadataMutationAOTOutput

  * [IN] Prepare for autograd: (nothing)
  * [IN] Create joint: TangentAOTInput (potentially w/
    IntermediateBaseAOTOutput, InputMutationAOTOutput)
  * [IN] Precompile subclass: SubclassGetAttrAOTInput et al.
  * [IN] Effect tokens: ForwardTokenAOTInput, BackwardTokenAOTInput
    (Note: BackwardTokenAOTInput is technically generated not by a wrapper but
    actually done by token_discovery which implicitly adds extra arguments
    to the FX trace on-the-fly.)

* Trigger a trace with the modified inputs on the wrapper
* Output desc propagation (happens as we unwind from the user function call in trace)

  * [OUT] Begin with original calling convention: PlainAOTOutput
  * [OUT] Effect tokens: ForwardTokenAOTOutput, BackwardTokenAOTOutput
  * [OUT] Precompile subclass: SubclassGetAttrAOTOutput et al.
  * [OUT] Create joint: GradAOTOutput
  * [OUT] Prepare for autograd: InputMutationAOTOutput, IntermediateBaseAOTOutput
  * [OUT] Precompile synthetic base: MetadataMutationAOTOutput
  * [OUT] Precompile dedupe: (nothing)
"""

import dataclasses


# TODO: the is_* predicates are a little suspicious because (1) they're not
# used by anything and (2) they always report False even when a parameter got
# swizzled into a view base or deduped with a non-parameter.  It is pretty
# difficult to exercise these cases but it's not clear if you will write code
# that works correctly in those cases.


@dataclasses.dataclass(frozen=True)
class AOTInput:
    # [CLASS DOC] Abstract base of all input descriptors. Immutable (frozen
    # dataclass), hashable, no fields of its own. Subclasses implement expr()
    # and override the is_* predicates when they represent params/buffers/tangents.
    """Describes where an input from an AOTAutograd produced FX graph comes from"""

    def expr(self) -> str:
        # [FUNCTION DOC] Return a human-readable path string for this input.
        # Abstract: always raises NotImplementedError in the base class.
        raise NotImplementedError("Subclasses must implement expr()")

    def is_param(self) -> bool:
        # [FUNCTION DOC] Return whether this input is (or derives from) a parameter.
        # Base default is False.
        """True if this input is a parameter or derived from a parameter (e.g., subclass attr)"""
        return False

    def is_buffer(self) -> bool:
        # [FUNCTION DOC] Return whether this input is (or derives from) a buffer.
        # Base default is False.
        """True if this input is a buffer or derived from a buffer (e.g., subclass attr)"""
        return False

    def is_tangent(self) -> bool:
        # [FUNCTION DOC] Return whether this input is (or derives from) a tangent.
        # Base default is False.
        """True if this input is a tangent or derived from a tangent (e.g., subclass attr)"""
        return False


# Note: Currently, our typing discipline for differentiable versus not is not
# very good, so feel free to rely on runtime tests instead.


@dataclasses.dataclass(frozen=True)
class DifferentiableAOTInput(AOTInput):
    # [CLASS DOC] Marker subclass for tensor inputs that may be wrapped by
    # GradAOTOutput (i.e. have a gradient). No added state or behavior.
    """A subclass that classifies AOTInput that can be wrapped by GradAOTOutput"""


@dataclasses.dataclass(frozen=True)
class AOTOutput:
    # [CLASS DOC] Abstract base of all output descriptors. Immutable, hashable.
    # Subclasses implement expr(); is_grad() marks gradient outputs.
    """Describes where an output from an AOTAutograd produced FX graph will
    eventually be bundled into the final output"""

    def expr(self) -> str:
        # [FUNCTION DOC] Return a human-readable path string for this output.
        # Abstract: always raises NotImplementedError in the base class.
        raise NotImplementedError("Subclasses must implement expr()")

    def is_grad(self) -> bool:
        # [FUNCTION DOC] Return whether this output is (or derives from) a gradient.
        # Base default is False.
        """True if this output is a grad or derived from a grad (e.g., subclass attr)"""
        return False


@dataclasses.dataclass(frozen=True)
class DifferentiableAOTOutput(AOTOutput):
    # [CLASS DOC] Marker subclass for tensor outputs that may be wrapped by
    # TangentAOTInput. No added state or behavior.
    """A subclass that classifies AOTOutput that can be wrapped by TangentAOTInput"""


# ------------

# AOTInput

# ------------


@dataclasses.dataclass(frozen=True)
class ParamAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for a module parameter input; field target is the
    # parameter's fully qualified name.
    """The input is a parameter, whose FQN is target"""

    target: str

    def expr(self) -> str:
        # [FUNCTION DOC] Return a string naming a get-parameter lookup of the target FQN.
        return f"self.get_parameter({self.target!r})"

    def is_param(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def is_buffer(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False


@dataclasses.dataclass(frozen=True)
class BufferAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for a module buffer input; field target is the
    # buffer's fully qualified name.
    """The input is a buffer, whose FQN is target"""

    target: str

    def expr(self) -> str:
        # [FUNCTION DOC] Return a string naming a get-buffer lookup of the target FQN.
        return f"self.get_buffer({self.target!r})"

    def is_param(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def is_buffer(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True


@dataclasses.dataclass(frozen=True)
class DummyAOTInput(AOTInput):
    # [CLASS DOC] Placeholder input descriptor used when descriptor processing
    # is irrelevant; field idx is an arbitrary index.
    """In some circumstances, we want to call into a function that expects AOTInput, but
    we don't actually care about that logic (most typically, because some code is being used
    for both compile-time and run-time; AOTInput processing is not needed in this situation.
    Pass a dummy in this situation; but it is better to just have a version of the function
    that doesn't have this at all."""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return a placeholder name made of "__dummy" and idx.
        return f"__dummy{self.idx}"


@dataclasses.dataclass(frozen=True)
class PlainAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for a positional input of the original flat
    # callable; field idx is the position.
    """The input is a plain input, corresponding to a particular positional index.

    Note that AOTInput is always relative to a function with a *flat* calling convention,
    e.g., as accepted by `aot_module_simplified`.  There are some AOTAutograd APIs that
    flatten pytrees, and we don't record PyTree key paths from the flattening (but we
    could and should!)
    """

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return a string indexing the original args at idx.
        return f"args[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTInput(AOTInput):
    # [CLASS DOC] Descriptor for an inner tensor attribute (attr) of a tensor
    # subclass input described by base. Predicates delegate to base.
    """Subclass inputs get unpacked into their constituent pieces before going into an FX
    graph.  This tells you which particular attribute of the subclass this particular
    input corresponds to (of the 'base' originally subclass argument.)
    """

    base: AOTInput
    attr: str

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by an attribute access of attr.
        return f"{self.base.expr()}.{self.attr}"

    def is_param(self) -> bool:
        # [FUNCTION DOC] Delegate to the base descriptor.
        return self.base.is_param()

    def is_buffer(self) -> bool:
        # [FUNCTION DOC] Delegate to the base descriptor.
        return self.base.is_buffer()

    def is_tangent(self) -> bool:
        # [FUNCTION DOC] Delegate to the base descriptor.
        return self.base.is_tangent()


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTInput(AOTInput):
    # [CLASS DOC] Descriptor for the outer size SymInt at dimension idx of a
    # subclass input described by base.
    """Which subclass this particular outer size SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by a size query at dim idx.
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTInput(AOTInput):
    # [CLASS DOC] Descriptor for the outer stride SymInt at dimension idx of a
    # subclass input described by base.
    """Which subclass this particular outer stride SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by a stride query at dim idx.
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class ViewBaseAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for the shared autograd view base that replaces
    # several aliasing view inputs; base_of names one of the replaced inputs.
    """
    When multiple differentiable inputs are views of the same input, AOTAutograd will replace all of these
    views with a single input representing the base.  If this is undesirable, you can clone the views
    example inputs before passing them into AOTAutograd.

    TODO: In principle we could report ALL of the inputs who this is a base of.
    """

    base_of: AOTInput

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base_of expression followed by a base-attribute access.
        return f"{self.base_of.expr()}._base"


@dataclasses.dataclass(frozen=True)
class SyntheticBaseAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for a synthetic base built when aliasing inputs
    # had no differentiable view; base_of names one of the replaced inputs.
    """This is similar to ViewBaseAOTInput, but this happens when none of the views were differentiable, so
    we weren't able to get our hands on the true original view and constructed a synthetic one instead
    for the sake of autograd.
    """

    base_of: AOTInput

    def expr(self) -> str:
        # [FUNCTION DOC] Return a make-synthetic-base call string wrapping the base_of expression.
        return f"__make_synthetic_base({self.base_of.expr()})"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardSeedAOTInput(AOTInput):
    # [CLASS DOC] Singleton-style descriptor (no fields) for the Philox RNG seed input of the forward graph.
    """The seed for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the forward Philox seed.
        return "__philox_forward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardBaseOffsetAOTInput(AOTInput):
    # [CLASS DOC] Descriptor (no fields) for the Philox RNG base offset input of the forward graph.
    """The offset for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the forward Philox base offset.
        return "__philox_forward_base_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardSeedAOTInput(AOTInput):
    # [CLASS DOC] Descriptor (no fields) for the Philox RNG seed input of the backward graph.
    """The seed for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the backward Philox seed.
        return "__philox_backward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardBaseOffsetAOTInput(AOTInput):
    # [CLASS DOC] Descriptor (no fields) for the Philox RNG base offset input of the backward graph.
    """The offset for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the backward Philox base offset.
        return "__philox_backward_base_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTInput(AOTInput):
    # [CLASS DOC] Descriptor for the idx-th effect (world) token input of the forward graph.
    """The world token which is threaded through side-effectful operations"""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__forward_token" followed by idx.
        return f"__forward_token{self.idx}"


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTInput(AOTInput):
    # [CLASS DOC] Descriptor for the idx-th effect (world) token input of the backward graph.
    """The world token which is threaded through side-effectful operations, for backwards"""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__backward_token" followed by idx.
        return f"__backward_token{self.idx}"


# Technically the "output" here is redundant, tangents always correspond to
# outputs
# NB: this is marked differentiable as it /would/ be differentiable if we
# support double backwards, but we never generate this today because we
# don't support double backwards.
@dataclasses.dataclass(frozen=True)
class TangentAOTInput(DifferentiableAOTInput):
    # [CLASS DOC] Descriptor for the tangent (incoming gradient) input of the
    # joint graph; field output is the DifferentiableAOTOutput it is the tangent of.
    """An input to the joint graph representing the tangent of an output."""

    output: DifferentiableAOTOutput

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate construction: raises AssertionError if output is
        # not a DifferentiableAOTOutput.
        if not isinstance(self.output, DifferentiableAOTOutput):
            raise AssertionError(
                f"expected output to be DifferentiableAOTOutput, got {type(self.output)}"
            )

    def expr(self) -> str:
        # [FUNCTION DOC] Return an output-tangent call string wrapping the output expression.
        return f"__output_tangent({self.output.expr()})"

    def is_tangent(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True


# ------------

# AOTOutput

# ------------


@dataclasses.dataclass(frozen=True)
class PlainAOTOutput(DifferentiableAOTOutput):
    # [CLASS DOC] Descriptor for the idx-th output of the original flat callable.
    """A plain tensor output at position idx of the output tuple"""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return a string indexing the output tuple at idx.
        return f"output[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class InputMutationAOTOutput(DifferentiableAOTOutput):
    # [CLASS DOC] Descriptor for the new value of a mutated input (mutated_input),
    # returned from the graph so autograd can differentiate through the mutation.
    """The mutated value of an input tensor, returned so we can appropriately propagate autograd."""

    mutated_input: AOTInput

    def expr(self) -> str:
        # [FUNCTION DOC] Return an input-mutation call string wrapping the mutated input expression.
        return f"__input_mutation({self.mutated_input.expr()})"


@dataclasses.dataclass(frozen=True)
class IntermediateBaseAOTOutput(DifferentiableAOTOutput):
    # [CLASS DOC] Descriptor for the shared intermediate base returned in place
    # of several aliasing outputs; base_of names one contributing output.
    """An intermediate base of multiple outputs which alias each other.  We only report ONE of
    the outputs that contributed to this base"""

    base_of: "AOTOutput"

    def expr(self) -> str:
        # [FUNCTION DOC] Return an intermediate-base call string wrapping the base_of expression.
        return f"__intermediate_base({self.base_of.expr()})"


# TODO: it's a little dodgy this is differentiable lol, but we do generate
# these BEFORE autograd is handled
@dataclasses.dataclass(frozen=True)
class MetadataMutationAOTOutput(DifferentiableAOTOutput):
    # [CLASS DOC] Descriptor for an aliased argument whose metadata was mutated
    # (synthetic-base handling); field idx identifies which one.
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__aliased_arg_with_metadata_mutation" followed by idx.
        return f"__aliased_arg_with_metadata_mutation{self.idx}"


# NB: this is marked differentiable as it /would/ be differentiable if we
# support double backwards, but we never generate this today because we
# don't support double backwards.
@dataclasses.dataclass(frozen=True)
class GradAOTOutput(DifferentiableAOTOutput):
    # [CLASS DOC] Descriptor for the gradient output of the joint graph; field
    # grad_of is the DifferentiableAOTInput the gradient is for.
    """An output representing the computed gradient for a differentiable input, in the joint graph"""

    grad_of: DifferentiableAOTInput

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate construction: raises AssertionError if grad_of is
        # not a DifferentiableAOTInput.
        if not isinstance(self.grad_of, DifferentiableAOTInput):
            raise AssertionError(
                f"expected grad_of to be DifferentiableAOTInput, got {type(self.grad_of)}"
            )

    def expr(self) -> str:
        # [FUNCTION DOC] Return a grad call string wrapping the grad_of expression.
        return f"__grad({self.grad_of.expr()})"

    def is_grad(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedForwardOffsetAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor (no fields) for the updated Philox RNG offset returned by the forward graph.
    """The final offset from the functionalized RNG calls, forward only"""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the updated forward offset.
        return "__philox_updated_forward_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedBackwardOffsetAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor (no fields) for the updated Philox RNG offset returned by the backward graph.
    """The final offset from the functionalized RNG calls, backward only"""

    def expr(self) -> str:
        # [FUNCTION DOC] Return the constant name of the updated backward offset.
        return "__philox_updated_backward_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for the idx-th effect token output of the forward graph.
    """The world token output for side-effectful calls, returned so we cannot DCE it, forward only"""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__forward_token" followed by idx.
        # KNOWN ISSUE: identical string to ForwardTokenAOTInput.expr, so input and output are not distinguishable by expr alone.
        return f"__forward_token{self.idx}"


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for the idx-th effect token output of the backward graph.
    """The world token output for side-effectful calls, returned so we cannot DCE it, backward only"""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__backward_token" followed by idx.
        return f"__backward_token{self.idx}"


# These are seemingly symmetric with their AOTInput counterparts.  The way to
# think about it is that a subclass could be an input or an output, and they
# get exploded into plain tensors on the way in and out.  So we need
# descriptors for both.
@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for an output that will be repacked as inner
    # tensor attribute attr of a subclass output described by base.
    """This output will be bundled into a subclass at this location"""

    base: AOTOutput
    attr: str

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by an attribute access of attr.
        return f"{self.base.expr()}.{self.attr}"

    def is_grad(self) -> bool:
        # [FUNCTION DOC] Delegate to the base descriptor.
        return self.base.is_grad()


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for an output size value at dimension idx that will
    # be repacked into the subclass described by base.
    """This output size will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by a size query at dim idx.
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for an output stride value at dimension idx that
    # will be repacked into the subclass described by base.
    """This output stride will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return the base expression followed by a stride query at dim idx.
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class DummyAOTOutput(AOTOutput):
    # [CLASS DOC] Placeholder output descriptor used when descriptor
    # propagation is irrelevant; field idx is an arbitrary index.
    """For cases when you don't actually care about descriptor propagation, do not use under normal
    circumstances."""

    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return a placeholder name made of "__dummy" and idx.
        return f"__dummy{self.idx}"


@dataclasses.dataclass(frozen=True)
class SavedForBackwardsAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for the idx-th activation saved for backward (a forward graph output).
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__saved_for_backwards_" followed by idx.
        return f"__saved_for_backwards_{self.idx}"


# Note [Activations with no version counter checks in eager]
# In eager, when a tensor is saved for backward, the autograd engine
# generally performs version counter checks when grabbing the activation
# at backward time.
# One exception is in a custom autograd.Function: if the user stashes a tensor
# for use in the backward using `ctx.foo = foo`, rather than
# `ctx.save_for_backward(foo)`, then the autograd engine will not know
# to perform any checks.
# In torch.compile, we handle autograd by tracing through it ahead of time,
# and wrapping the fw + bw graphs into a custom autograd.Function.
# In order to provide parity with eager around VC checks, though, we need
# to know which activations had gotten this "no VC checks" treatment in eager,
# so we can plumb this info into the AOTAutograd runtime,
# and avoid calling ctx.save_for_backward on these tensors.
#
# This dataclass tells us that a given AOTOutput corresponds to:
# - An activation output from the forward graph
# - that should *not* have its version counter checked at runtime in the backward.
#   This is done by stashing this activation on the ctx object directly
@dataclasses.dataclass(frozen=True)
class SavedForBackwardsNoVcCheckAOTOutput(AOTOutput):
    # [CLASS DOC] Descriptor for the idx-th forward-graph activation that must be
    # stashed directly on the autograd ctx (no save_for_backward, so no version
    # counter check at backward time), matching eager custom Function behavior.
    idx: int

    def expr(self) -> str:
        # [FUNCTION DOC] Return "__saved_for_backwards_no_vc_check_" followed by idx.
        return f"__saved_for_backwards_no_vc_check_{self.idx}"
