# anonymity_loss_coefficient
Contains the python code for measuring the Anonymity Loss Coefficient

## To install

`pip install anonymity-loss-coefficient`

## Usage

The file `example.py` contains an example of how to use `anonymity_loss_coefficient`. 

The file `example.md` is the output of `example.py`.

## Dev

To push to pypi.org:

remove the `dist/` directory

Update the version in `setup.py`

`python setup.py sdist bdist_wheel`

`twine upload dist/*`