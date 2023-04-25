"""Top-level package for efficientbioai acceleration."""

__author__ = "mmv_lab team"
__email__ = "yu.zhou@isas.de"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.5"


def get_module_version():
    return __version__


from .compress_ppl import Pipeline  # noqa: F401,E402
