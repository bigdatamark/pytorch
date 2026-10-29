# [PACKAGE DOC] Experimental functorch API namespace. Re-exports chunk_vmap,
# replace_all_batch_norm_modules_, hessian, jacfwd, jvp and functionalize from torch's
# functorch implementation. Importing has no side effects beyond those imports.
# PyTorch forward-mode is not mature yet
from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
from torch.func import functionalize
