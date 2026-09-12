"""Regression tests for yahoo_fantasy_bot-3o9: declared dependency metadata
must actually match what the code imports, and setup.py/requirements.txt
must not contradict each other.

These tests parse the two files as plain text/AST rather than importing
setup.py (which would call setup() as a side effect), so they're cheap and
don't require the package to be installed.
"""
import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / 'requirements.txt'
SETUP_PY_PATH = REPO_ROOT / 'setup.py'

REQUIRED_FOR_SCORING = {'pandas', 'numpy', 'openpyxl'}


def _requirements_names():
    names = set()
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = re.split(r'[=<>!~\[]', line, maxsplit=1)[0].strip()
        names.add(name.lower())
    return names


def _setup_call_kwargs():
    """Parse setup.py's AST and return the setup(...) call's keyword args as
    a dict of {kwarg_name: ast node}, without executing the file."""
    tree = ast.parse(SETUP_PY_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'setup':
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError('could not find a setup(...) call in setup.py')


def _install_requires_names(kwargs):
    install_requires = kwargs['install_requires']
    names = set()
    for elt in install_requires.elts:
        value = elt.value
        name = re.split(r'[=<>!~\[]', value, maxsplit=1)[0].strip()
        names.add(name.lower())
    return names


def _yahoo_fantasy_api_spec(names_with_specifiers):
    for spec in names_with_specifiers:
        if spec.lower().startswith('yahoo_fantasy_api'):
            return spec
    return None


def _raw_requirements_lines():
    return [
        line.strip() for line in REQUIREMENTS_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


def _raw_install_requires(kwargs):
    return [elt.value for elt in kwargs['install_requires'].elts]


def test_requirements_txt_declares_scoring_dependencies():
    names = _requirements_names()
    missing = REQUIRED_FOR_SCORING - names
    assert not missing, (
        f'requirements.txt is missing packages the scoring/ranking code '
        f'imports: {sorted(missing)}'
    )


def test_setup_py_install_requires_declares_scoring_dependencies():
    kwargs = _setup_call_kwargs()
    names = _install_requires_names(kwargs)
    missing = REQUIRED_FOR_SCORING - names
    assert not missing, (
        f'setup.py install_requires is missing packages the scoring/ranking '
        f'code imports: {sorted(missing)}'
    )


def test_setup_py_declares_rapidfuzz_as_an_optional_extra():
    kwargs = _setup_call_kwargs()
    assert 'extras_require' in kwargs, (
        'setup.py must declare extras_require for the optional rapidfuzz '
        'fuzzy-matching dependency'
    )
    extras_dict = kwargs['extras_require']
    extras = {}
    for key_node, value_node in zip(extras_dict.keys, extras_dict.values):
        extras[key_node.value] = [elt.value for elt in value_node.elts]
    fuzzy_deps = [d for group in extras.values() for d in group
                  if d.lower().startswith('rapidfuzz')]
    assert fuzzy_deps, (
        f'expected an extras_require group containing rapidfuzz, got: {extras}'
    )
    # rapidfuzz must NOT be a hard (always-installed) requirement, since the
    # code only imports it lazily/optionally.
    names = _install_requires_names(kwargs)
    assert 'rapidfuzz' not in names


def test_yahoo_fantasy_api_version_is_reconciled_between_the_two_files():
    kwargs = _setup_call_kwargs()
    setup_spec = _yahoo_fantasy_api_spec(_raw_install_requires(kwargs))
    req_spec = _yahoo_fantasy_api_spec(_raw_requirements_lines())
    assert setup_spec is not None and req_spec is not None
    assert setup_spec == req_spec, (
        f'setup.py ({setup_spec!r}) and requirements.txt ({req_spec!r}) '
        f'must declare the same yahoo_fantasy_api version constraint'
    )
    # Prefer a >= floor over an exact pin (per the fix direction), and no
    # pin the codebase can't justify.
    assert '==' not in setup_spec, (
        f'yahoo_fantasy_api should use a >= floor, not an exact pin: '
        f'{setup_spec!r}'
    )


def test_setup_py_python_requires_and_classifier_match_actual_python_support():
    kwargs = _setup_call_kwargs()
    python_requires = kwargs['python_requires'].value
    # The code uses f-strings (3.6+) and .travis.yml targets 3.7; must not
    # claim support for anything older than that.
    assert re.match(r'>=\s*3\.(?:[89]|1\d)', python_requires), (
        f'python_requires={python_requires!r} should be >=3.8 (f-strings '
        f'need 3.6+, and CI targets 3.7+)'
    )

    classifiers_node = kwargs['classifiers']
    classifiers = [elt.value for elt in classifiers_node.elts]
    py_classifiers = [c for c in classifiers
                       if c.startswith('Programming Language :: Python :: 3.')]
    assert py_classifiers, 'expected a specific Python 3.x classifier'
    for c in py_classifiers:
        version = c.rsplit('::', 1)[-1].strip()
        major, minor = (int(x) for x in version.split('.'))
        assert (major, minor) >= (3, 7), (
            f'classifier {c!r} claims support for a Python version older '
            f'than the code actually supports (f-strings need 3.6+, CI '
            f'targets 3.7+)'
        )
