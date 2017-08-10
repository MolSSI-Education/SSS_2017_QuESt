#! /usr/bin/env python
"""
This is a setup script for install a python quest with a CMake dependency.

For simplicity this will use Psi4's Cache to make detection extremely simple
"""

import setuptools
from setuptools.command.install import install

import os
import subprocess as sp
import shutil

try:
    import psi4
except ImportError:
    raise ImportError("Cannot find Psi4. Please make sure Psi4 is installed so"
                      "that this setup script can detect Psi4's CMake dependency trees.\n"
                      "Please run the following conda command: 'conda install psi4 -c psi4'")


def sanitize_cmake(output):
    print_out = []

    # Cut out a few warnings that are a bit annoying, GCC wont let us override them
    warnings = [
        ' warning: section "__textcoal_nt"', 'note: change section name to "__const"',
        'change section name to "__text"', 'note: change section name to "__data"',
        ' warning: section "__datacoal_nt"', 'warning: section "__const_coal"'
    ]
    x = 0
    while x < len(output):

        if any(warn in output[x] for warn in warnings):
            x += 3
            continue

        print_out.append(output[x])
        x += 1

    print_out = "\n".join(print_out)
    return print_out


class cmake_build(install):

    #    description = 'Build the nested CMake quest'

    def run(self):

        # Find build directory (in-place)
        abspath = os.path.abspath(os.path.dirname(__file__))
        build_path = os.path.join(abspath, "quest", "core")
        os.chdir(build_path)
        print(">>> cd {}".format(build_path))

        # Run cmake command
        print("Build CMake temporaries...")
        print(">>> cmake .\n")
        output = sp.check_output(["cmake", "."]).decode("UTF-8")
        print(output)
        if "Build files have been written to" not in output:
            raise Exception("Psi4 Cache Error.\n")

        # Run install
        print("Compiling...")
        output = sp.check_output(["make", "-j2", "VERBOSE=1"], stderr=sp.STDOUT).decode("UTF-8").splitlines()
        print_out = sanitize_cmake(output)
        #print_out = '\n'.join(output)
        print(">>> make -j2\n{}".format(print_out))

        if "[100%]" not in print_out:
            raise Exception("Build error.\n")


class cmake_clean(install):
    def run(self):

        # Find build directory (in-place)
        abspath = os.path.abspath(os.path.dirname(__file__))
        build_path = os.path.join(abspath, "quest", "core")
        os.chdir(build_path)
        print("Removing CMake build files...")

        try:
            shutil.rmtree("CMakeFiles")
        except:
            pass

        files = ["CMakeCache.txt", "Makefile", "timer.dat", "cmake_install.cmake", "core.cpython-35m-darwin.so"]
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

        print("...finished")

if __name__ == "__main__":
    setuptools.setup(
        name='quest',
        version="0.2.1",
        description='A hybrid C++/Python program for QM and MM',
        author='The MolSSI 2017 Summer School',
        author_email='sss@molssi.org',
        url="https://github.com/molssi-sss/SSS_2017_Project",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
            'pytest>=3.0',
            'pytest-cov',
            'pyyaml',
            'matplotlib',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
        cmdclass={
            'cmake': cmake_build,
            'clean': cmake_clean,
        },
    )
