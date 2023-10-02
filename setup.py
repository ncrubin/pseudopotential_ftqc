"""
package for mec_sandia

"""

import io
import re

from setuptools import setup


def version_number(path: str) -> str:
    """Get S0's version number from the src directory
    """
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, 'r') as fqe_version:
        version = version_re.search(fqe_version.read()).group(1)

    return version


def main() -> None:
    """
    """
    version_path = 'pseudopotential_ftqc/_version.py'

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError('Version information not found in ' + version_path)

    long_description = ('=======\n' +
                        'GOOG-COVESTRO COLLAB.\n' +
                        '=======\n')
    stream = io.open('README.md', encoding='utf-8')
    stream.readline()
    long_description += stream.read()

    requirements_buffer = open('requirements.txt').readlines()
    requirements = [r.strip() for r in requirements_buffer]

    setup(
        name='pseudopotential_ftqc',
        version=__version__,
        author='Folks',
        author_email='rubinnc0@gmail.com',
        description='Folks-RUS',
        long_description=long_description,
        install_requires=None,
        license='Apache 2',
        packages=["pseudopotential_ftqc"],
        )


main()