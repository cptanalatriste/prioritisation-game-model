# Equilibrium Analysis for Bug Reporting
This collection of Python scripts make use of a simulation model of bug reporting for obtaining the Nash-Equilibrium of its empirical game-theoretic Model.

## Pre-requisites
The code was built and tested using the [Anaconda distribution for Python 2.7](https://www.continuum.io/DOWNLOADS). For equilibrium calculation, it relies on the [Gambit Software version 15](http://www.gambit-project.org/).

Besides the libraries available in Anaconda, we also need the following:
- *Simpy* version 2.3.1, for the simulation model implementation.
- *Pathos* version 0.2.0, for the parallel execution of simulation replications.
- *Recordtype*, for mutable named tuples support.

## Configuration
The file `gtconfig.py` exposes the configurations parameters. Edit following according to your needs:

- `all_issues_csv`, is the file location of the CSV file containing the bug tracking system data needed for obtaining the simulation inputs.
- `gambit_folder`, is the installation directory of the Gambit Software
- `quantal_response_solver`, is the name of the command for obtaining the quantal response equilibria in Gambit.
- `replications_per_profile`, is the number of replications to execute per strategy profile. 
- `parallel`, to enable the parallel execution of simulation replications.
- `parallel_blocks`, the number of parallel blocks to divide simulation execution. It can be set to the number of cores available on your system.
- `is_windows`, should be `False` if you are not using a Windows operating system.

## Execution
The search-based mechanism design experiments are based on games with the following characteristics:

- *Simulation model:* Bug reports arrive individually to a priority queue based on the priority contained in the report.
- *Strategy catalog:* Contains only the always-honest and always-dishonest strategies.
- *Game reduction:* No game reduction is performed, nor any symmetry assumption is made.
- *Player selection:* From the bug tracking system data, we pick the most productive bug reporters as players.
- *Equilibrium algorithm:* Quantal response equilibrium, since is the one recommended by Gambit for games with more than 2 players.

To execute this scenario, you can do the following: 

`python getequilibrium.py testers developers target_bugs  file_name`

Where:

- *testers* is the number of bug reporters.
- *developers* is the number of developers available for bug fixing.
- *target_bugs* is the number of bugs to be fixed before release.
- *file_name* is the CSV file name where the equilibrium will be stored.
