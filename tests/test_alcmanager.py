import os
import shutil
import random
import pandas as pd
import logging
import pytest
from typing import Tuple, Optional
from anonymity_loss_coefficient import ALCManager

debug = False

def guess_with_prob(value: int, param1: Optional[float], param2: Optional[float]) -> Tuple[int, bool]:
    # using random, select a random number between 0 and 1
    prob_correct = param1
    prob_abstain = param2
    ran1 = random.random()
    ran2 = random.random()
    if ran1 < prob_correct:
        guess = value
    else:
        guess = 1 - value
    if ran2 < prob_abstain:
        abstain = True
    else:
        abstain = False
    return guess, abstain

def make_df() -> pd.DataFrame:
    """
    Creates a DataFrame with two columns, c1 and c2, and 4000 rows.
    Each combination of (c1, c2) values (0,0), (0,1), (1,0), and (1,1)
    appears exactly 1000 times.

    Returns:
        pd.DataFrame: The constructed DataFrame.
    """
    data = {
        "c1": [0] * 2000 + [1] * 2000,
        "c2": [0] * 1000 + [1] * 1000 + [0] * 1000 + [1] * 1000
    }
    # shuffle the DataFrame to randomize the order of rows
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

@pytest.fixture(scope="session")
def temp_dir():
    """
    Fixture to create a temporary directory for test results.
    The directory is created once per test session and deleted after all tests are done.
    """
    dir_path = "tests/temp_dir"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path  # Provide the directory path to the tests
    # Teardown: Remove the directory and its contents after all tests
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    if not debug:
        shutil.rmtree(dir_path)

@pytest.fixture
def df():
    """Fixture to create and return the DataFrame."""
    return make_df()

@pytest.mark.parametrize(
    "my_func, param1, param2, expected_alc",
    [
        (guess_with_prob, 1.0, 0.9, 0.87),  # All guesses correct, 90% of runs = abstain
        (guess_with_prob, 1.0, 0.0, 1.0),  # All guesses correct, no abstain
        (guess_with_prob, 0.0, 0.0, -0.9),  # All guesses wrong, no abstain
        (guess_with_prob, 0.5, 0.0, 0.0),  # Half of guesses correct, no abstain
    ]
)
def test_basic(temp_dir, df, my_func, param1, param2, expected_alc):
    """
    Runs the basic_test for each parameterized condition.
    """
    # Initialize ALCManager
    alcm = ALCManager(df, df.copy(), results_path=temp_dir, flush=True, random_state=42)

    # Run predictions
    for _, secret_value, _ in alcm.predictor(known_columns=['c1'], secret_column='c2'):
        predicted_value, abstain = my_func(secret_value, param1, param2)
        if abstain:
            alcm.abstention()
        else:
            alcm.prediction(predicted_value, 1.0)
        if debug:
            import pprint
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(alcm.halt_info)

    # Get results
    if debug:
        pp.pprint(alcm.halt_info)
        alcm.summarize_results()
    df_grouped = alcm.alc_per_secret_and_known_df(known_columns=['c1'], secret_column='c2')
    alcm.close_logger()
    df_grouped = df_grouped[df_grouped['paired'] == False]
    print("---------------------")
    for column in df_grouped.columns:
        print(f"{column}: {df_grouped.iloc[0][column]}")
    base_prec = df_grouped.iloc[0]['base_prec']
    # Base precision should always be around 0.05 for this dataset
    assert base_prec == pytest.approx(0.5, abs=0.05), f"Expected base precision 0.5, got {base_prec}"

    base_si_low = df_grouped.iloc[0]['base_si_low']
    base_si_high = df_grouped.iloc[0]['base_si_high']
    # assert that base_si_high - base_si_low is less than 0.1
    assert base_si_high - base_si_low < 0.1, f"Expected base SI range to be less than 0.1, got {base_si_high - base_si_low}"
    attack_si_low = df_grouped.iloc[0]['attack_si_low']
    attack_si_high = df_grouped.iloc[0]['attack_si_high']
    assert attack_si_high - attack_si_low < 0.1, f"Expected attack SI range to be less than 0.1, got {attack_si_high - attack_si_low}"

    alc = df_grouped.iloc[0]['alc']
    # Assert the ALC value is close to the expected value
    assert alc == pytest.approx(expected_alc, abs=0.1), f"Expected ALC: {expected_alc}, but got: {alc}"