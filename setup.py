from setuptools import setup, find_packages

setup(
    name='anonymity_loss_coefficient',
    version='1.0.27',
    description='A package to run attacks on anonymized data using the Anonymity Loss Coefficient.',
    author='Paul Francis',
    author_email='paul@francis.com',
    url='https://github.com/yoid2000/anonymity_loss_coefficient',
    packages=find_packages(include=["anonymity_loss_coefficient", "anonymity_loss_coefficient.*"]),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyarrow>=20.0.0",
        "fastparquet>=2024.1.0",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)