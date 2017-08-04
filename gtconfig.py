import logging

git_home = "E:\OneDrive\phd2\jira_github_ds"
all_issues_csv = git_home + "\\apache_jira_github_ds.csv"

gambit_folder = "C:\Program Files (x86)\Gambit\\"
enumerate_equilibria_solver = "gambit-enummixed.exe"
quantal_response_solver = "gambit-logit.exe"

replications_per_profile = 200
report_stream_batching = True
simple_reporting_model = False

parallel = True

parallel_blocks = 4

do_gatekeeper = True
success_rates = [1.0, 0.9]

do_throttling = False
inflation_factors = [0.01, 0.03, 0.05]

is_windows = True

# This parameters are only used in the experiments module (penaltyexp)
use_empirical_strategies = True
use_heuristic_strategies = True


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
