# Effects of curvature, gradient and path width on cyclist behaviour

This repository contains the code for the thesis completed as part of my master's degree
at KTH.

## Getting started

To create the data used in the thesis and the graphs presented for the three locations:
1. Replace the stub files in the ``data`` folder.
2. Run the code in the [main.py](thesis/main.py) file.

For more detail, check out the following files and the functions contained in them:

* The code in [preprocessing](thesis/preprocessing/__init__.py) processes the source files,
converts them into a different format and extracts some information from them.

* The code in [processing](thesis/processing/__init__.py) extracts
additional information from the trajectories (e.g. overtakes) and saves the results.

* The [report](thesis/results/report) package includes everything included in
the final report. Most of the filtering is also done there. The code here is a bit messier than the other packages,
but documentation is included for most functions.

## License
The code in this repository is licensed under the [GPLv3 licence](LICENSE).