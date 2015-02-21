# Neural Encoding Models

A set of python modules for fitting, testing, and visualizing parameters of neural encoding models (NEMs).

Neural encoding models are models that try and predict neural activity given a stimulus. For example, we can fit models to predict the spiking activity of neurons in the retina or V1 in response to a visual stimulus displayed on a computer monitor.

We include general tools that allow you to fit the parameters of encoding models of any functional form. Additionally, we provide specific classes to fit linear-nonlinear (LN) and cascaded (2-layer) linear-nonlinear (LN-LN) models to data.

## Installation
```bash
git clone git@github.com:ganguli-lab/nems.git
cd nems
pip install -r requirements.txt
python setup.py install
```

## Requirements
Numpy, scipy, pandas and the [proxalgs](https://github.com/ganguli-lab/proxalgs) package.

## Development
Pull requests welcome! Please stick to the [NumPy/SciPy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard)
We use `sphinx` for documentation and `nose` for testing.

## Todo
- clean up LNLN model class
- add an LN model class
- add some examples or a walkthrough
- flesh out the documentation
- develop some unit tests

## Contact
Niru Maheswaranathan (nirum@stanford.edu)