
<p align="center">
<a href="https://travis-ci.org/MolSSI-SSS/SSS_2017_QuESt">
  <img src="https://travis-ci.org/MolSSI-SSS/SSS_2017_QuESt.svg?branch=master" alt="Travis CI"/>
</a>
<a href="https://codecov.io/gh/MolSSI-SSS/SSS_2017_QuESt">
  <img src="https://codecov.io/gh/MolSSI-SSS/SSS_2017_QuESt/branch/master/graph/badge.svg" alt="Codecov" />
</a>
</p>

# QuESt: Quantum Energy and Stuff 
A hybrid QM/MM project built on the principles of the MolSSI 2017 software
summer school.

## Installation
To install you first need the MolSSI 2017 Software Summer School stack.
Directions can be found
[here](https://molssi-sss.github.io/Logistics_SSS_2017/Setup.html).

To build the C++ side of this project, please run:
```
python setup.py cmake
```

To clean the CMake files,
```
python setup.py clean
```

To do a local install of the Python directory,
```
pip install -e .
```


## Testing
Tests can be run using `py.test -v`.

