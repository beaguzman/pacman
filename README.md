# Installation

### Prerequisites

Ensure that you're using Python version 3.12. If you have a different version, you might experience GUI-related issues. Check your Python version by running:
```
python --version
```

### Installing

After installing Miniconda, set up your environment with the following commands:
```
conda create --name cs221-hw5 python=3.12
conda activate cs221-hw5
This homework does not require any additional packages, so feel free to reuse the conda environment you installed earlier for previous homeworks.
```

# Pacman instructions

Pacman extra credit should be uploaded as a separate gradescope assignment. To
do so, note grader_ec.py is the same as grader.py but adds only question 4 to
the extra credit. Run `publish-assignment.py` one with the original grader.py
to make the main programming assignment, and once with `grader_ec.py`
overwriting `grader.py` to make the extra credit assignment.

# Other instructions

To play, type:   python pacman.py

For help, type:  python pacman.py -h

See http://inst.eecs.berkeley.edu/~cs188 for more information.