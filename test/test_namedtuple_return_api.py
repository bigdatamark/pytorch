# Owner(s): ["module: unknown"]

# [MODULE DOC] Tests that operators returning several tensors expose named tuple results.
# Holds an allowlist of operator names permitted to have named returns in the native function
# schema file, a skip list of attention operators, and one test class. The module reads the
# schema yaml file located relative to this test file at import time only to compute its path.
import os
import re
import yaml
import textwrap
import torch

from torch.testing._internal.common_utils import TestCase, run_tests, HardwareClassification
from collections import namedtuple


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
all_operators_with_namedtuple_return = {
    'max', 'min', 'aminmax', 'median', 'nanmedian', 'mode', 'kthvalue', 'svd',
    'geqrf', 'slogdet', 'sort', 'topk', 'linalg_inv_ex',
    'triangular_solve', 'cummax', 'cummin', 'linalg_eigh', "_linalg_eigh", "_unpack_dual", 'linalg_qr',
    'linalg_svd', '_linalg_svd', 'linalg_slogdet', '_linalg_slogdet', 'fake_quantize_per_tensor_affine_cachemask',
    'fake_quantize_per_channel_affine_cachemask', 'linalg_lstsq', 'linalg_eig', 'linalg_cholesky_ex',
    'frexp', 'lu_unpack', 'histogram', 'histogramdd',
    '_fake_quantize_per_tensor_affine_cachemask_tensor_qparams',
    '_fused_moving_avg_obs_fq_helper', 'linalg_lu_factor', 'linalg_lu_factor_ex', 'linalg_lu',
    '_linalg_det', '_lu_with_info', 'linalg_ldl_factor_ex', 'linalg_ldl_factor', 'linalg_solve_ex', '_linalg_solve_ex',
    'linalg_polar'
}

all_operators_with_namedtuple_return_skip_list = {
    '_scaled_dot_product_flash_attention',
    '_scaled_dot_product_fused_attention_overrideable',
    '_scaled_dot_product_flash_attention_for_cpu',
    '_scaled_dot_product_efficient_attention',
    '_scaled_dot_product_cudnn_attention',
}


class TestNamedTupleAPI(TestCase):
    # [CLASS DOC] Test case checking the named tuple return API: the torch.return_types module,
    # the schema file's naming rules, and field access by name for each allowlisted operator.
    # The class attribute hw_classification marks the test as generic (hardware independent).
    hw_classification = HardwareClassification.GENERIC

    def test_import_return_types(self):
        # [FUNCTION DOC] Checks that the torch.return_types module can be imported and that a
        # star import from it succeeds. Raises if either import fails; no other assertions.
        import torch.return_types  # noqa: F401
        exec('from torch.return_types import *')

    def test_native_functions_yaml(self):
        # [FUNCTION DOC] Parses the native functions yaml schema file and checks that only
        # allowlisted operators declare named return values. Entries whose names are in the
        # allowlist are recorded and skipped; backward and forward-suffixed names, non-tuple or
        # empty returns, and skip-listed operators are ignored; any other tuple return element
        # with more than one token (type plus name) fails. Finally asserts the recorded set equals
        # the allowlist, so a renamed or removed operator is reported.
        operators_found = set()
        regex = re.compile(r"^(\w*)(\(|\.)")
        with open(aten_native_yaml) as file:
            for f in yaml.safe_load(file.read()):
                f = f['func']
                ret = f.split('->')[1].strip()
                name = regex.findall(f)[0][0]
                if name in all_operators_with_namedtuple_return :
                    operators_found.add(name)
                    continue
                if '_backward' in name or name.endswith('_forward'):
                    continue
                if not ret.startswith('('):
                    continue
                if ret == '()':
                    continue
                if name in all_operators_with_namedtuple_return_skip_list:
                    continue
                ret = ret[1:-1].split(',')
                for r in ret:
                    r = r.strip()
                    self.assertEqual(len(r.split()), 1, 'only allowlisted '
                                     'operators are allowed to have named '
                                     'return type, got ' + name)
        self.assertEqual(all_operators_with_namedtuple_return, operators_found, textwrap.dedent("""
        Some elements in the `all_operators_with_namedtuple_return` of test_namedtuple_return_api.py
        could not be found. Do you forget to update test_namedtuple_return_api.py after renaming some
        operator?
        """))

    def test_namedtuple_return(self):
        # [FUNCTION DOC] For each entry in a table of operator names, expected field names, extra
        # positional inputs and an out-variant flag, checks that results are named tuples.
        # Three checks per operator when available: calling the torch (or torch.linalg or
        # torch._VF) function, calling its out= variant, and calling the Tensor method; each
        # result must expose fields matching tuple positions, and a matching class must exist in
        # torch.return_types and be constructible. Finally asserts the covered operator names
        # equal the allowlist, so a missing table entry is reported.
        a = torch.randn(5, 5)
        per_channel_scale = torch.randn(5)
        per_channel_zp = torch.zeros(5, dtype=torch.int64)

        op = namedtuple('op', ['operators', 'input', 'names', 'hasout'])
        operators = [
            op(operators=['max', 'min', 'median', 'nanmedian', 'mode', 'sort', 'topk', 'cummax', 'cummin'], input=(0,),
               names=('values', 'indices'), hasout=True),
            op(operators=['kthvalue'], input=(1, 0),
               names=('values', 'indices'), hasout=True),
            op(operators=['svd'], input=(), names=('U', 'S', 'V'), hasout=True),
            op(operators=['linalg_svd', '_linalg_svd'], input=(), names=('U', 'S', 'Vh'), hasout=True),
            op(operators=['slogdet', 'linalg_slogdet'], input=(), names=('sign', 'logabsdet'), hasout=True),
            op(operators=['_linalg_slogdet'], input=(), names=('sign', 'logabsdet', 'LU', 'pivots'), hasout=True),
            op(operators=['linalg_qr'], input=(), names=('Q', 'R'), hasout=True),
            op(operators=['linalg_polar'], input=(), names=('U', 'H'), hasout=True),
            op(operators=['geqrf'], input=(), names=('a', 'tau'), hasout=True),
            op(operators=['triangular_solve'], input=(a,), names=('solution', 'cloned_coefficient'), hasout=True),
            op(operators=['linalg_eig'], input=(), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['linalg_eigh'], input=("L",), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['_linalg_eigh'], input=("L",), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['linalg_cholesky_ex'], input=(), names=('L', 'info'), hasout=True),
            op(operators=['linalg_inv_ex'], input=(), names=('inverse', 'info'), hasout=True),
            op(operators=['linalg_solve_ex'], input=(a,), names=('result', 'info'), hasout=True),
            op(operators=['_linalg_solve_ex'], input=(a,), names=('result', 'LU', 'pivots', 'info'), hasout=True),
            op(operators=['linalg_lu_factor'], input=(), names=('LU', 'pivots'), hasout=True),
            op(operators=['linalg_lu_factor_ex'], input=(), names=('LU', 'pivots', 'info'), hasout=True),
            op(operators=['linalg_ldl_factor'], input=(), names=('LD', 'pivots'), hasout=True),
            op(operators=['linalg_ldl_factor_ex'], input=(), names=('LD', 'pivots', 'info'), hasout=True),
            op(operators=['linalg_lu'], input=(), names=('P', 'L', 'U'), hasout=True),
            op(operators=['fake_quantize_per_tensor_affine_cachemask'],
               input=(0.1, 0, 0, 255), names=('output', 'mask',), hasout=False),
            op(operators=['fake_quantize_per_channel_affine_cachemask'],
               input=(per_channel_scale, per_channel_zp, 1, 0, 255),
               names=('output', 'mask',), hasout=False),
            op(operators=['_unpack_dual'], input=(0,), names=('primal', 'tangent'), hasout=False),
            op(operators=['linalg_lstsq'], input=(a,), names=('solution', 'residuals', 'rank', 'singular_values'), hasout=False),
            op(operators=['frexp'], input=(), names=('mantissa', 'exponent'), hasout=True),
            op(operators=['lu_unpack'],
               input=(torch.tensor([3, 2, 1, 4, 5], dtype=torch.int32), True, True),
               names=('P', 'L', 'U'), hasout=True),
            op(operators=['histogram'], input=(1,), names=('hist', 'bin_edges'), hasout=True),
            op(operators=['histogramdd'], input=(1,), names=('hist', 'bin_edges'), hasout=False),
            op(operators=['_fake_quantize_per_tensor_affine_cachemask_tensor_qparams'],
               input=(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int), torch.tensor([1]), 0, 255),
               names=('output', 'mask',), hasout=False),
            op(operators=['_fused_moving_avg_obs_fq_helper'],
               input=(torch.tensor([1]), torch.tensor([1]), torch.tensor([0.1]), torch.tensor([0.1]),
               torch.tensor([0.1]), torch.tensor([1]), 0.01, 0, 255, 0), names=('output', 'mask',), hasout=False),
            op(operators=['_linalg_det'],
               input=(), names=('result', 'LU', 'pivots'), hasout=True),
            op(operators=['aminmax'], input=(), names=('min', 'max'), hasout=True),
            op(operators=['_lu_with_info'],
               input=(), names=('LU', 'pivots', 'info'), hasout=False),
        ]

        def get_func(f):
            # [FUNCTION DOC] HELPER: Resolves an operator name string to a callable. Names with
            # a linalg prefix map to the torch.linalg module with the prefix removed; names with a
            # leading underscore map to torch._VF. Returns None if the attribute is absent.
            "Return either torch.f or torch.linalg.f, where 'f' is a string"
            mod = torch
            if f.startswith('linalg_'):
                mod = torch.linalg
                f = f[7:]
            if f.startswith('_'):
                mod = torch._VF
            return getattr(mod, f, None)

        def check_namedtuple(tup, names):
            # [FUNCTION DOC] HELPER: Asserts each named field of the given tuple is the same
            # object as the element at the matching position.
            "Check that the namedtuple 'tup' has the given names"
            for i, name in enumerate(names):
                self.assertIs(getattr(tup, name), tup[i])

        def check_torch_return_type(f, names):
            # [FUNCTION DOC] HELPER: Looks up the return type class for the operator name in
            # torch.return_types, builds it from a list of scalar tensors, one per field name, and
            # asserts the constructed object has that class.
            """
            Check that the return_type exists in torch.return_types
            and they can constructed.
            """
            return_type = getattr(torch.return_types, f)
            inputs = [torch.randn(()) for _ in names]
            self.assertEqual(type(return_type(inputs)), return_type)

        for op in operators:
            for f in op.operators:
                # 1. check the namedtuple returned by calling torch.f
                func = get_func(f)
                if func:
                    ret1 = func(a, *op.input)
                    check_namedtuple(ret1, op.names)
                    check_torch_return_type(f, op.names)
                #
                # 2. check the out= variant, if it exists
                if func and op.hasout:
                    ret2 = func(a, *op.input, out=tuple(ret1))
                    check_namedtuple(ret2, op.names)
                    check_torch_return_type(f + "_out", op.names)
                #
                # 3. check the Tensor.f method, if it exists
                meth = getattr(a, f, None)
                if meth:
                    ret3 = meth(*op.input)
                    check_namedtuple(ret3, op.names)

        all_covered_operators = {x for y in operators for x in y.operators}

        self.assertEqual(all_operators_with_namedtuple_return, all_covered_operators, textwrap.dedent('''
        The set of covered operators does not match the `all_operators_with_namedtuple_return` of
        test_namedtuple_return_api.py. Do you forget to add test for that operator?
        '''))

if __name__ == '__main__':
    run_tests()
