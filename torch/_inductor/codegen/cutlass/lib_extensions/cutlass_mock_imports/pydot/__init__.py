# [PACKAGE DOC] HELPER: Mock of the pydot package; only defines Dot as None so imports of it succeed
# without the real dependency.
# mypy: disable-error-code="var-annotated"
Dot = None
