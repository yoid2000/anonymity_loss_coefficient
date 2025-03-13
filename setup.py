from setuptools import setup, find_packages

setup(
    name='anonymity_loss_coefficient',
    version='1.0.0',
    description='A package to calculate the anonymity loss coefficient (ALC).',
    author='Paul Francis',
    author_email='paul@francis.com',
    url='https://github.com/yoid2000/anonymity_loss_coefficient',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)