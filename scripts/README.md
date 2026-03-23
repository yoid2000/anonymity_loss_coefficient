# scripts

This directory contains scripts that run the attacks in `anonymity_loss_coefficient/attacks`.

The scripts for each attack are placed in directories that correspond to the directories in `anonymity_loss_coefficient/attacks`.

This directory also contains `generic_example.py`, which contains the code used to generate `generic_example.md`. This is an example of how to build a new attack. If called with the --temp_dir command line parameter, then it runs without generating a results directory. Otherwise, it places the results in `./generic_example_files`.