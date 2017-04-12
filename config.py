git_home = "D:\git"
all_issues_csv = git_home + "\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED.csv"

gambit_folder = "C:\Program Files (x86)\Gambit\\"
enumerate_equilibria_solver = "gambit-enummixed.exe"
quantal_response_solver = "gambit-logit.exe"

replications_per_profile = 1000
use_empirical_strategies = True
use_heuristic_strategies = True

parallel = True
parallel_blocks = 4

do_gatekeeper = True
success_rates = [1.0, 0.9]

do_throttling = False
inflation_factors = [0.01, 0.03, 0.05]

is_windows = True
