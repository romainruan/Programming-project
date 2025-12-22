# -*- coding: utf-8 -*-
import math
import pulp
import matplotlib.pyplot as plt

# --- 1. DATA GENERATION ---
def get_data():
    """
    Returns a dictionary of venues and a pre-calculated distance matrix.
    Using 'Depart' as the starting point (node 0).
    """
    venues = {
        0: {"name": "Start", "x": 0, "y": 0, "stay": 0, "cost": 0, "satisfaction": 0},
        1: {"name": "Bar A", "x": 2, "y": 1, "stay": 1.2, "cost": 7, "satisfaction": 4.5},
        2: {"name": "Bar B", "x": 3, "y": 4, "stay": 0.8, "cost": 5, "satisfaction": 3.8},
        3: {"name": "Bar C", "x": 6, "y": 1, "stay": 1.5, "cost": 8, "satisfaction": 4.8},
        4: {"name": "Bar D", "x": 7, "y": 5, "stay": 1.0, "cost": 6, "satisfaction": 4.2},
        5: {"name": "Bar E", "x": 9, "y": 2, "stay": 1.3, "cost": 9, "satisfaction": 5.0},
    }
    
    distances = {}
    for i in venues:
        for j in venues:
            if i != j:
                # Euclidean distance calculation
                distances[i, j] = round(math.hypot(venues[i]['x']-venues[j]['x'], 
                                                   venues[i]['y']-venues[j]['y']), 2)
    return venues, distances

# --- 2. OPTIMIZATION MODEL (ILP) ---
def solve_barathon(venues, distances, max_time=15, max_budget=50):
    """
    Solves the Orienteering Problem using Integer Linear Programming.
    """
    model = pulp.LpProblem("Barathon_Optimization", pulp.LpMaximize)
    nodes = list(venues.keys())
    
    # Decision Variables
    # x[i] = 1 if venue i is visited
    x = pulp.LpVariable.dicts("visit", nodes, cat=pulp.LpBinary)
    # y[i,j] = 1 if travel occurs from venue i to j
    y = pulp.LpVariable.dicts("route", ((i, j) for i in nodes for j in nodes if i != j), cat=pulp.LpBinary)
    # u[i] = continuous variable for MTZ subtour elimination
    u = pulp.LpVariable.dicts("order", nodes, lowBound=0, upBound=len(nodes)-1, cat=pulp.LpContinuous)

    # Objective Function: Maximize total satisfaction
    model += pulp.lpSum(venues[i]["satisfaction"] * x[i] for i in nodes)

    # Constraint 1: Total Time (Travel Time + Stay Time)
    model += pulp.lpSum(distances[i, j] * y[i, j] for i, j in y) + \
             pulp.lpSum(venues[i]["stay"] * x[i] for i in nodes) <= max_time

    # Constraint 2: Total Budget
    model += pulp.lpSum(venues[i]["cost"] * x[i] for i in nodes) <= max_budget

    # Constraint 3: Flow Conservation (Entry = Exit = Visited)
    for i in nodes:
        model += pulp.lpSum(y[i, j] for j in nodes if i != j) == x[i]
        model += pulp.lpSum(y[j, i] for j in nodes if i != j) == x[i]

    # Constraint 4: MTZ Subtour Elimination
    for i, j in y:
        if i != 0 and j != 0:
            model += u[i] - u[j] + len(nodes) * y[i, j] <= len(nodes) - 1

    # Solve the model quietly
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        return 0, [], x
        
    # Extract the edges that form the optimal path
    optimal_path = [edge for edge, var in y.items() if var.value() > 0.9]
    return pulp.value(model.objective), optimal_path, x

# --- 3. SENSITIVITY ANALYSIS ---
def run_sensitivity_analysis(venues, distances):
    """
    Evaluates how the total satisfaction evolves with different time limits.
    """
    time_steps = range(5, 26, 2) # From 5h to 25h
    scores = []
    
    for t in time_steps:
        score, _, _ = solve_barathon(venues, distances, max_time=t)
        scores.append(score)
    
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, scores, marker='s', linestyle='-', color='navy')
    plt.title("Sensitivity Analysis: Satisfaction vs. Available Time")
    plt.xlabel("Total Time Limit (hours)")
    plt.ylabel("Total Satisfaction Score")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

# --- 4. VISUALIZATION ---
def plot_route(venues, path_edges, title):
    """
    Plots the venues and the directed edges of the optimal route.
    """
    plt.figure(figsize=(10, 6))
    
    # Draw venues
    for i, v in venues.items():
        color = 'darkorange' if i == 0 else 'royalblue'
        plt.scatter(v['x'], v['y'], s=250, c=color, zorder=3, edgecolors='black')
        plt.text(v['x']+0.2, v['y']+0.2, f"{v['name']}\n(S:{v['satisfaction']})", 
                 fontsize=9, fontweight='bold')

    # Draw the path with arrows
    for i, j in path_edges:
        plt.annotate("", xy=(venues[j]['x'], venues[j]['y']), 
                     xytext=(venues[i]['x'], venues[i]['y']),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5, mutation_scale=20))
    
    plt.title(title, fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize Data
    venues_data, dist_matrix = get_data()

    # 1. Run optimization for the baseline scenario (T=15h)
    score_opt, best_path, _ = solve_barathon(venues_data, dist_matrix, max_time=15)
    
    print(f"Optimal Solution Found!")
    print(f"Total Satisfaction Score: {score_opt}")
    print(f"Route: {' -> '.join([f'{i}' for i, j in best_path])} -> End")

    # 2. Plot the spatial results
    plot_route(venues_data, best_path, f"Optimal Venue Route (Time=15h, Score={score_opt})")

    # 3. Perform and plot sensitivity analysis
    run_sensitivity_analysis(venues_data, dist_matrix)