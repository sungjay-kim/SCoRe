"""Make the repository importable regardless of the directory it was cloned into.

Modules here address each other as ``corepp.metrics_3d``, ``corepp.networks`` and so on,
a holdover from when this code lived inside a folder named ``corepp``. That only resolves
if the checkout happens to carry that name and its parent is on ``sys.path``; a plain
``git clone`` produces ``SCoRe/`` and every entry point fails on its first import.

Importing this module puts the repository root on ``sys.path`` and binds it under the name
``corepp`` as well, so both spellings reach the same packages whatever the folder is
called. Entry points do this before their own imports::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import _repo_bootstrap  # noqa: F401
"""
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if 'corepp' not in sys.modules:
    _pkg = types.ModuleType('corepp')
    _pkg.__path__ = [str(ROOT)]
    _pkg.__doc__ = 'Alias for the repository root, kept for the corepp.* import paths.'
    sys.modules['corepp'] = _pkg
