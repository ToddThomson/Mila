"""
Supplies the version. Everything else is declarative in pyproject.toml.

The version is derived from the repository's Version.txt by the CMake build (see
cmake/MilaVersion.cmake and Mila/Adaptors/CMakeLists.txt) into the generated VERSION file
beside this one, rather than hand-copied into pyproject.toml -- where the binding's copy
silently drifted 18 builds behind. A missing VERSION means this tree was never configured,
which is an error rather than a default: guessing here would publish a wrong number, and a
published version can never be reused.

Unlike the binding's setup.py there is no BinaryDistribution here. MIS is pure Python and
py3-none-any is the correct tag -- it depends on the binary wheel, it does not contain one.
"""

from pathlib import Path

from setuptools import setup


def read_version() -> str:
    version_file = Path(__file__).parent / "VERSION"

    if not version_file.is_file():
        raise SystemExit(
            f"{version_file} is missing. It is generated when CMake configures the "
            "repository, which derives it from Version.txt -- configure the build before "
            "packaging."
        )

    return version_file.read_text(encoding="utf-8").strip()


setup(version=read_version())
