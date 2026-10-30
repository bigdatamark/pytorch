# [MODULE DOC] Loading helpers for the DLMC sparse dataset used by the sparse benchmarks. The
# dataset files describe only matrix structure (shape, nonzero count, row pointers, column
# indices); these helpers read the structure and fill the values with random numbers, generate
# dense vectors and matrices, and provide generators yielding operand pairs for matrix-vector,
# matrix-matrix and general matmul benchmarks, including gradient setup and scipy copies. Reads
# files and prints progress to standard output.
import math
from pathlib import Path

from scipy import sparse

import torch


def to_coo_scipy(x):
    # [FUNCTION DOC] Converts a torch sparse coo tensor to a scipy coo matrix using its internal
    # indices and values converted to numpy arrays (so the tensor must be on cpu and in a layout
    # those accessors support). Returns the scipy matrix with the same shape.
    indices_1 = x._indices().numpy()
    values_1 = x._values().numpy()
    return sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])), shape=x.shape)


def sparse_grad_output(a, b):
    # [FUNCTION DOC] Builds a random gradient for the sparse matrix product of the two operands.
    # Computes the product; if sparse, returns a random dense tensor masked to the sparsity
    # pattern of the coalesced product (a sparse tensor); otherwise returns a random tensor
    # shaped like the product.
    c = torch.sparse.mm(a, b)
    if c.is_sparse:
        c2 = torch.rand_like(c.to_dense())
        return c2.sparse_mask(c.coalesce())
    else:
        return torch.rand_like(c)


def read_matrix_params(path):
    # [FUNCTION DOC] Reads the first line of a dataset file, which holds row count, column count
    # and nonzero count separated by comma and space. Returns a pair: the shape tuple and the
    # nonzero count. Raises a value error if the line is malformed.
    with open(path) as file:
        line = file.readline()
        nrows, ncols, nnz = (int(el) for el in line.split(", "))
        return (nrows, ncols), nnz


def csr_to_coo(indices, indptr, shape):
    # [FUNCTION DOC] Expands compressed-row storage into coordinate form: given column indices,
    # row pointer array and shape, assigns each stored entry its row number by walking the row
    # ranges. Returns a long tensor with two rows (row indices and column indices).
    n_rows, n_cols = shape
    cols = indices
    rows = [0] * len(cols)
    for i in range(n_rows):
        for j in range(indptr[i], indptr[i + 1]):
            rows[j] = i
    return torch.tensor([rows, cols], dtype=torch.long)


def load_sparse_matrix(path, device):
    # [FUNCTION DOC] Reads a dataset file and returns a sparse coo tensor of double precision on
    # the given device. The first line gives shape and nonzero count, the second the row
    # pointers, the third the column indices; the stored values are not read from the file but
    # drawn from a standard normal distribution.
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = torch.randn(nnz, dtype=torch.double)
    shape = (nrows, ncols)
    return torch.sparse_coo_tensor(
        csr_to_coo(indices, index_pointers, shape), data, shape, device=device
    )


def gen_vector(path, device):
    # [FUNCTION DOC] Reads only the first line of the dataset file for the row count and returns
    # a random double-precision vector with that many entries on the given device.
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        return torch.randn(nrows, dtype=torch.double, device=device)


def gen_matrix(path, device):
    # [FUNCTION DOC] Reads only the first line of the dataset file for the shape and returns a
    # random double-precision dense matrix of that shape on the given device.
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        return torch.randn(nrows, ncols, dtype=torch.double, device=device)


def load_spmv_dataset(dataset_path, hidden_size, sparsity, device, n_limit=math.inf):
    # [FUNCTION DOC] Generator for sparse-matrix by dense-vector operands. Searches the dataset
    # folder for the given sparsity recursively for matrix files, scanning at most the limit
    # number of files, and prints the arguments and a dot per file. Files whose column count
    # equals the hidden size are collected as matrices; files whose row count equals the hidden
    # size are collected as vector sources. The two lists are paired by position, and each pair
    # yields a tuple of the sparse matrix loaded from one file and a random vector sized from
    # the other.
    """load_spmv_dataset loads a DLMC dataset for a sparse matrix-vector multiplication (SPMV) performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        device:
            Whether to place the Tensor on a GPU or CPU.
        n_limit:
            This value allows a dataset with some limit size.
    """
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob("**/*.smtx")
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    for f in files:
        if index >= n_limit:
            break
        print(".", end="")
        size, nnz = read_matrix_params(f.as_posix())
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        index += 1  # noqa: SIM113
    print()

    for fx, fy in zip(x_files, y_files):
        x = load_sparse_matrix(fx, device)
        y = gen_vector(fy, device)
        yield (x, y)


def load_spmm_dataset(
    dataset_path, hidden_size, sparsity, spmm_type, device, n_limit=math.inf
):
    # [FUNCTION DOC] Generator for sparse-matrix by matrix operands. Scans the files exactly as
    # the vector variant does (matching column count and row count against the hidden size,
    # positional pairing, limit on files scanned, progress output). Each pair yields a tuple of
    # the sparse matrix from one file and either a random dense matrix (when the type is the
    # sparse by dense product) or a sparse matrix loaded from the other file.
    """load_spmm_dataset loads a DLMC dataset for a sparse matrix-matrix multiplication (SPMM) performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        spmm_type:
            This value allows tensors for `sparse@sparse` or `sparse@dense` operations.
        device:
            Whether to place the Tensor on a GPU or CPU.
        n_limit:
            This value allows a dataset with some limit size.
    """
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob("**/*.smtx")
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    for f in files:
        if index >= n_limit:
            break
        print(".", end="")
        size, nnz = read_matrix_params(f.as_posix())
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        index += 1  # noqa: SIM113
    print()

    for fx, fy in zip(x_files, y_files):
        x = load_sparse_matrix(fx, device)
        y = (
            gen_matrix(fy, device)
            if spmm_type == "sparse@dense"
            else load_sparse_matrix(fy, device)
        )
        yield (x, y)


def load_dlmc_dataset(
    dataset_path,
    operation,
    hidden_size,
    sparsity,
    device,
    requires_grad,
    n_limit=math.inf,
):
    # [FUNCTION DOC] Generator yielding one variable mapping per operand pair for the chosen
    # operation, delegating to the matrix or vector loader. On cpu the mapping includes scipy
    # copies of both operands. Without gradients it holds dense copies of the operands; with
    # gradients it also holds a sparse and a dense random output gradient, marks the operands as
    # requiring gradients, and uses detached dense clones that require gradients. Each mapping
    # has the sparse operands, the dense operands, and the optional scipy and gradient entries.
    # For an unrecognized operation the loader is never bound and an unbound-name error is
    # raised at the first use.
    """load_dlmc_dataset loads a DLMC dataset for a matmul performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        operation:
            This value allows tensors for `sparse@sparse`|`sparse@dense`|`sparse@vector` operations.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        device:
            Whether to place the Tensor on a GPU or CPU.
        requires_grad:
            Loads the dataset for backward test.
        n_limit:
            This value allows a dataset with some limit size.
    """
    if operation == "sparse@sparse" or operation == "sparse@dense":
        collection = load_spmm_dataset(
            dataset_path, hidden_size, sparsity, operation, device, n_limit
        )
    elif operation == "sparse@vector":
        collection = load_spmv_dataset(
            dataset_path, hidden_size, sparsity, device, n_limit
        )
    scipy_vars = {}
    backward_vars = {}
    for x, y in collection:
        if device == "cpu":
            scipy_vars = {
                "sx": to_coo_scipy(x) if x.is_sparse else x.numpy(),
                "sy": to_coo_scipy(y) if y.is_sparse else y.numpy(),
            }
        if not requires_grad:
            dx = x.to_dense() if x.is_sparse else x
            dy = y.to_dense() if y.is_sparse else y
        else:
            c = sparse_grad_output(x, y)
            backward_vars = {
                "sparse_grad_output": c,
                "grad_output": c.to_dense() if c.is_sparse else c,
            }
            x.requires_grad_(True)
            y.requires_grad_(True)
            dx = x.to_dense().detach() if x.is_sparse else x.clone().detach()
            dy = y.to_dense().detach() if y.is_sparse else y.clone().detach()
            dx.requires_grad_(True)
            dy.requires_grad_(True)
        yield {"x": x, "y": y, "dx": dx, "dy": dy, **scipy_vars, **backward_vars}
