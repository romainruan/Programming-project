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
│ └── synthetic_data.py # Fixed venues and distances
│
├── models/
│ ├── ilp_model.py # ILP optimization model
│ └── baseline.py # Greedy baseline heuristic
│
├── experiments/
│ ├── main.py # Main script (runs ILP, baseline, sensitivity analysis)
│ └── plot_path.py # Visualization of the optimal route
```
