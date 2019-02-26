import logging
import os

git_home = "/home/carlos/Documents/datasets"
all_issues_csv = git_home + "/apache_jira_github_ds.csv"

gambit_folder = "/home/carlos/gambit-15.1.1/"
enumerate_equilibria_solver = "gambit-enummixed"
quantal_response_solver = "gambit-logit"

report_stream_batching = True
simple_reporting_model = False

fix_count_criteria = True  # True for ending simulation after a number of fixes. False to use the development time budget.
parallel = True  # Set to False for debugging purposes
parallel_blocks = 4

# Equilibrium experiment configurations. Used by payoffgetter.py and penaltyexp.py
priority_queues = [True]
dev_team_factors = [0.5, 1.0]
exclude_self_fix = False
replications_per_profile = 40
use_empirical_strategies = True  # Disable for quick experimentation
use_heuristic_strategies = True

# Experiment configuration for Gatekeeper. Used by penaltyexp.py
do_gatekeeper = True
success_rates = [0.5, 0.9, 1.0]

# Experiment configuration for Throttling. Used by penaltyexp.py
do_throttling = False
inflation_factors = [0.01, 0.03, 0.05, 0.07]

is_windows = (os.name == 'nt')
beep = True


def get_logger(name="gtbugreporting", filename="gtbugreporting.log", level=logging.INFO):
    """
    Returns a logger instance, for file logging.
    :param name: Name of the module running.
    :param filename: File to log.
    :return: Logging instance.
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    file_handler = logging.FileHandler("logs/" + filename, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(level)

    return logger
