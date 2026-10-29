# mypy: disable-error-code=attr-defined
# [PACKAGE DOC] Vendored, trimmed copy of the "unification" library used by the
# FX experimental type/shape machinery. Provides first-order unification of
# nested tuples/lists/dicts/objects containing logic variables, plus reification
# (substituting a substitution back into a term). Re-exports reify and unify
# (core), unifiable (class registration helper), and the logic-variable API
# isvar, Var, var, variables, vars (variable module).
from .core import reify, unify
from .more import unifiable
from .variable import isvar, Var, var, variables, vars
