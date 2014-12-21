NGCluster
=========

NGCluster is an experiment with clustering gene expression data using
neighborhood graphs.

Ben Saylor and Londen Johnson  
Dr. Rahul Singh, Professor, Computer Science, SFSU  
CSC 857 - Bioinformatics Computing, Fall 2014  
San Francisco State University

Software Dependencies
---------------------

The easiest way to satisfy all of the dependencies is to install the [Anaconda
Python distribution](http://continuum.io/downloads#py34).

- Python 3 (version 3.4.0 has been tested)
- LLVM (version 3.3 has been tested)
  - See the [llvmpy web site](http://www.llvmpy.org/#quickstart) for
    installation instructions
- NumPy (version 1.9.1 has been tested)
- SciPy library (version 0.14.0 has been tested)
- Matplotlib (version 1.4.2 has been tested)
  - If you encounter a TypeError when installing Matplotlib, the workaround
    described [here](https://stackoverflow.com/a/27085321) may apply.
- llvmpy (version 0.12.7 has been tested)
- Numba (version 0.15.1 has been tested)

All but the first two packages are installable from PyPI and can be installed
using `pip install -r requirements.txt`.

Usage
-----

NGCluster includes a number of configurations of graph and clustering
parameters. Running a configuration produces a set of output files in a
directory named output/*configuration-key*, where *configuration-key* is the
short name for one of the configurations listed by `python run.py`. Running the
program also compiles summary output for the listed and all previously run
configurations in output/compiled-results.csv.

To list available configurations:

    python run.py

To run a set of configurations:

    python run.py configuration-key-1 [configuration-key-2 ...]

To run all configurations:

    python run.py all

To run no configurations but produce the compiled output file based on existing
output:

    python run.py compile
