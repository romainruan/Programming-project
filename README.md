# Programming Project

## Required packages
- pulp
- matplotlib
- numpy

## Project structure
```
Project/
│
├── data/
│ └── fixed_data.py # Fixed (non-random) venues and distances
│
├── models/
│ ├── ilp_model.py # ILP optimization model
│ └── baseline.py # Greedy baseline heuristic
│
├── experiments/
│ ├── run_experiments.py # Main script (runs ILP, baseline, sensitivity analysis)
│ └── plot_path.py # Visualization of the optimal route
```
