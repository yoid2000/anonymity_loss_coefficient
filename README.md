# anonymity_loss_coefficient

This code contains:

1. Code to support developing attacks on anonymized data using the Anonymity Loss Coefficient.
2. A set of attacks.
3. Example code for running the attacks.

For users only interested in running existing attacks, the `scripts` directory contains those attacks. Please see `scripts/README.md` for more information.

The directory `anonymity_loss_coefficient/alc` contains the code that is used by attacks to produce the ALC measures. The `ALCManager` class in `anonymity_loss_coefficient/alc/alc_manager.py` contains the API for access to this functionality.

The directory `anonymity_loss_coefficient/attacks` contains the attacks themselves, one sub-directory per attack. These use the `ALCManager` class.

For users interested in developing new attacks, `scripts/generic_example.md` outlines the basic usage of the `ALCManager` to run attacks. The attacks themselves also serve as examples.

## To install

`pip install anonymity-loss-coefficient`

## Dev

To push to pypi.org:

remove the `dist/` directory

Update the version in `setup.py`

`python setup.py sdist bdist_wheel`

`twine upload dist/*`