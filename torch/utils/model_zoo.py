# [MODULE DOC] Backward-compatibility shim for the old model zoo location.
# Re-exports tqdm, and load_url (an alias of the torch.hub function load_state_dict_from_url)
# so that existing imports such as torchvision keep working. No logic of its own.
# torchvision imports tqdm from here.
from torch.hub import tqdm, load_state_dict_from_url as load_url  # noqa: F401
