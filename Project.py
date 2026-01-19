"""
BARATHON OPTIMIZATION
====================================================================
Authors: Ruben Salaün, Romain Ruan, Jules Nguyen
Institution: Uvic, Barcelona
"""

import math
import random
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Venue:
    """Represents a bar/venue with all relevant attributes"""
    id: int
    name: str
    x: float
    y: float
    lat: float
    lon: float
    stay_time: float
    cost: float
    satisfaction: float
    opening_time: float
    closing_time: float
    category: str
    
    def __hash__(self):
        return hash(self.id)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_realistic_barcelona_data(n_venues: int = 10) -> Tuple[Dict[int, Venue], np.ndarray]:
    """Generates realistic bar data inspired by Barcelona's nightlife."""
    
    LAT_MIN, LAT_MAX = 41.38, 41.40
    LON_MIN, LON_MAX = 2.15, 2.19
    
    categories = {
        'cerveceria': {'stay': (0.5, 1.0), 'cost': (8, 15), 'sat': (6.0, 8.0)},
        'cocktail_bar': {'stay': (0.8, 1.5), 'cost': (12, 20), 'sat': (7.0, 9.0)},
        'wine_bar': {'stay': (0.6, 1.2), 'cost': (10, 18), 'sat': (6.5, 8.5)},
        'pub': {'stay': (0.5, 1.0), 'cost': (8, 14), 'sat': (6.0, 7.5)},
        'tapas_bar': {'stay': (0.7, 1.3), 'cost': (15, 25), 'sat': (7.5, 9.5)},
    }
    
    bar_names = [
        "La Cervecita", "El Raval Cocktails", "Bodega Montserrat", "The Irish Pub",
        "Paradiso", "Cal Pep", "Bobby's Free", "Satan's Coffee Corner", 
        "Marula Cafe", "Jamboree"
    ]
    
    venues = {}
    
    # Starting point
    venues[0] = Venue(
        id=0, name="Home", x=0, y=0, lat=41.3851, lon=2.1734,
        stay_time=0, cost=0, satisfaction=0, opening_time=0, 
        closing_time=24, category='start'
    )
    
    # Generate bars with more compact distances
    for i in range(1, n_venues + 1):
        cat = random.choice(list(categories.keys()))
        cat_info = categories[cat]
        
        # Keep bars closer together (within 2km radius)
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.3, 2.0)  # km
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        lat = venues[0].lat + y / 110.54
        lon = venues[0].lon + x / (111.32 * math.cos(math.radians(lat)))
        
        venues[i] = Venue(
            id=i,
            name=bar_names[i-1] if i <= len(bar_names) else f"Bar {i}",
            x=x, y=y, lat=lat, lon=lon,
            stay_time=round(random.uniform(*cat_info['stay']), 1),
            cost=round(random.uniform(*cat_info['cost']), 1),
            satisfaction=round(random.uniform(*cat_info['sat']), 1),
            opening_time=18.0,  # Simplified: all open at 6pm
            closing_time=2.0,   # Simplified: all close at 2am
            category=cat
        )
    
    # Calculate distance matrix (walking time in hours)
    n = len(venues)
    distances = np.zeros((n, n))
    
    for i in venues:
        for j in venues:
            if i != j:
                dx = venues[i].x - venues[j].x
                dy = venues[i].y - venues[j].y
                dist_km = math.sqrt(dx**2 + dy**2)
                # Walking speed: 5 km/h
                distances[i][j] = round(dist_km / 5.0, 3)
    
    return venues, distances

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_path_time(path: List[int], venues: Dict[int, Venue], 
                       distances: np.ndarray) -> float:
    """Calculates total time for a given path."""
    total_time = 0
    for i in range(len(path) - 1):
        total_time += distances[path[i]][path[i+1]]
        if path[i] != 0:
            total_time += venues[path[i]].stay_time
    return total_time

def print_solution_details(sol: Dict, venues: Dict[int, Venue], distances: np.ndarray):
    """Prints detailed solution information."""
    print(f"\n{'='*70}")
    print(f"  {sol['algorithm'].upper()} SOLUTION")
    print(f"{'='*70}")
    print(f"Status: {sol['status']}")
    print(f"Total Satisfaction: {sol['objective']:.1f} points")
    print(f"Number of Venues Visited: {sol['n_venues']}")
    print(f"Total Time: {sol['total_time']:.2f} hours ({sol['total_time']*60:.0f} minutes)")
    print(f"Total Cost: €{sol['total_cost']:.2f}")
    print(f"\nRoute Details:")
    print(f"{'-'*70}")
    
    path = sol['path']
    cumulative_time = 0
    
    for idx in range(len(path)):
        venue_id = path[idx]
        venue = venues[venue_id]
        
        if venue_id == 0:
            if idx == 0:
                print(f"  START: {venue.name}")
            else:
                print(f"  END: Return to {venue.name}")
                print(f"  Final cumulative time: {cumulative_time:.2f}h")
        else:
            travel_time = distances[path[idx-1]][venue_id] if idx > 0 else 0
            cumulative_time += travel_time
            
            print(f"\n  {idx}. {venue.name} ({venue.category})")
            print(f"     • Travel time: {travel_time*60:.1f} min")
            print(f"     • Arrival at: {cumulative_time:.2f}h")
            print(f"     • Stay time: {venue.stay_time*60:.0f} min")
            print(f"     • Cost: €{venue.cost}")
            print(f"     • Satisfaction: {venue.satisfaction}/10")
            
            cumulative_time += venue.stay_time
    
    print(f"{'='*70}\n")

# ============================================================================
# ALGORITHM 1: GREEDY HEURISTIC
# ============================================================================

def solve_greedy_routing_aware(venues: Dict[int, Venue],
                               distances: np.ndarray,
                               max_time: float = 6.0,
                               max_budget: float = 100.0,
                               start_time: float = 20.0) -> Dict:
    """Greedy algorithm that considers travel time and efficiency ratio."""
    
    current_pos = 0
    current_time = 0
    current_budget = 0
    path = [0]
    unvisited = set(venues.keys()) - {0}
    
    while unvisited and current_budget < max_budget and current_time < max_time:
        best_venue = None
        best_score = -float('inf')
        
        for venue_id in unvisited:
            venue = venues[venue_id]
            
            travel_time = distances[current_pos][venue_id]
            total_time_cost = current_time + travel_time + venue.stay_time
            total_budget_cost = current_budget + venue.cost
            
            # Check constraints
            if total_time_cost > max_time or total_budget_cost > max_budget:
                continue
            
            # Calculate efficiency score
            time_cost = travel_time + venue.stay_time
            resource_cost = time_cost + venue.cost / 20  # normalize cost
            efficiency = venue.satisfaction / max(resource_cost, 0.1)
            
            if efficiency > best_score:
                best_score = efficiency
                best_venue = venue_id
        
        if best_venue is None:
            break
        
        current_time += distances[current_pos][best_venue] + venues[best_venue].stay_time
        current_budget += venues[best_venue].cost
        path.append(best_venue)
        current_pos = best_venue
        unvisited.remove(best_venue)
    
    # Return home
    path.append(0)
    current_time += distances[current_pos][0]
    total_satisfaction = sum(venues[i].satisfaction for i in path if i != 0)
    
    return {
        'algorithm': 'Greedy',
        'status': 'Feasible',
        'objective': total_satisfaction,
        'path': path,
        'visited': [i for i in path if i != 0],
        'total_time': current_time,
        'total_cost': current_budget,
        'n_venues': len(path) - 2
    }

# ============================================================================
# ALGORITHM 2: 2-OPT LOCAL SEARCH
# ============================================================================

def solve_two_opt_local_search(venues: Dict[int, Venue],
                               distances: np.ndarray,
                               max_time: float = 6.0,
                               max_budget: float = 100.0,
                               max_iterations: int = 500) -> Dict:
    """2-opt local search starting from greedy solution."""
    
    initial_sol = solve_greedy_routing_aware(venues, distances, max_time, max_budget)
    current_path = initial_sol['path'][:-1]
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, len(current_path) - 1):
            for j in range(i + 1, len(current_path)):
                new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                
                time_cost = calculate_path_time(new_path + [0], venues, distances)
                budget_cost = sum(venues[k].cost for k in new_path if k != 0)
                
                if time_cost <= max_time and budget_cost <= max_budget:
                    new_satisfaction = sum(venues[k].satisfaction for k in new_path if k != 0)
                    old_satisfaction = sum(venues[k].satisfaction for k in current_path if k != 0)
                    
                    if new_satisfaction > old_satisfaction:
                        current_path = new_path
                        improved = True
                        break
            if improved:
                break
    
    final_path = current_path + [0]
    total_time = calculate_path_time(final_path, venues, distances)
    total_cost = sum(venues[i].cost for i in final_path if i != 0)
    total_satisfaction = sum(venues[i].satisfaction for i in final_path if i != 0)
    
    return {
        'algorithm': '2-Opt',
        'status': 'Feasible',
        'objective': total_satisfaction,
        'path': final_path,
        'visited': [i for i in final_path if i != 0],
        'total_time': total_time,
        'total_cost': total_cost,
        'n_venues': len(final_path) - 2,
        'iterations': iteration
    }

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(venues: Dict[int, Venue],
                            distances: np.ndarray) -> pd.DataFrame:
    """Runs sensitivity analysis on time and budget parameters."""
    
    results = []
    
    print("\n Running sensitivity analysis...")
    print("   Testing different time limits...")
    for time_limit in np.arange(2, 8, 0.5):
        sol = solve_greedy_routing_aware(venues, distances, max_time=time_limit)
        results.append({
            'parameter': 'time',
            'value': time_limit,
            'algorithm': 'Greedy',
            'satisfaction': sol['objective'],
            'n_venues': sol['n_venues'],
            'cost': sol['total_cost']
        })
    
    print("   Testing different budget limits...")
    for budget in range(40, 151, 10):
        sol = solve_greedy_routing_aware(venues, distances, max_budget=budget)
        results.append({
            'parameter': 'budget',
            'value': budget,
            'algorithm': 'Greedy',
            'satisfaction': sol['objective'],
            'n_venues': sol['n_venues'],
            'cost': sol['total_cost']
        })
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(venues: Dict[int, Venue],
                         distances: np.ndarray,
                         solutions: Dict[str, Dict],
                         sensitivity_df: pd.DataFrame):
    """Creates publication-quality visualizations."""
    
    # Figure 1: Route Map
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get best solution
    best_algo = max(solutions.keys(), key=lambda k: solutions[k]['objective'])
    sol = solutions[best_algo]
    path = sol['path']
    
    # Plot all venues
    for i, venue in venues.items():
        if i == 0:
            ax.scatter(venue.x, venue.y, s=500, c='red', marker='*', 
                      zorder=5, edgecolors='black', linewidths=2, label='Start/End')
        elif i in path:
            ax.scatter(venue.x, venue.y, s=300, c='green', 
                      zorder=4, edgecolors='black', linewidths=2, alpha=0.8, label='Visited' if i == path[1] else '')
        else:
            ax.scatter(venue.x, venue.y, s=150, c='lightgray', 
                      zorder=3, alpha=0.5, label='Not visited' if i == 1 and i not in path else '')
    
    # Plot route with arrows
    for i in range(len(path) - 1):
        v1, v2 = venues[path[i]], venues[path[i+1]]
        ax.annotate('', xy=(v2.x, v2.y), xytext=(v1.x, v1.y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='blue', alpha=0.6))
        
        # Add step number
        mid_x = (v1.x + v2.x) / 2
        mid_y = (v1.y + v2.y) / 2
        ax.text(mid_x, mid_y, str(i+1), fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor='blue'))
    
    # Add venue labels
    for i in path[1:-1]:
        v = venues[i]
        ax.text(v.x + 0.15, v.y + 0.15, f"{v.name}\n€{v.cost} | {v.satisfaction}/10", 
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_title(f"Best Barathon Route ({best_algo})\n"
                f"Total Satisfaction: {sol['objective']:.1f} | "
                f"Venues: {sol['n_venues']} | "
                f"Time: {sol['total_time']:.1f}h | "
                f"Cost: €{sol['total_cost']:.1f}",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Distance East-West (km)', fontsize=12)
    ax.set_ylabel('Distance North-South (km)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('barathon_route.png', dpi=300, bbox_inches='tight')
    print("Saved: barathon_route.png")
    plt.show()
    
    # Figure 2: Sensitivity Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time sensitivity
    time_data = sensitivity_df[sensitivity_df['parameter'] == 'time']
    axes[0].plot(time_data['value'], time_data['satisfaction'], 
                marker='o', linewidth=3, markersize=8, color='#2E86AB')
    axes[0].fill_between(time_data['value'], time_data['satisfaction'], alpha=0.3, color='#2E86AB')
    axes[0].set_xlabel('Time Limit (hours)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Total Satisfaction Score', fontsize=13, fontweight='bold')
    axes[0].set_title('Impact of Time Constraint', fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Budget sensitivity
    budget_data = sensitivity_df[sensitivity_df['parameter'] == 'budget']
    axes[1].plot(budget_data['value'], budget_data['satisfaction'], 
                marker='s', linewidth=3, markersize=8, color='#A23B72')
    axes[1].fill_between(budget_data['value'], budget_data['satisfaction'], alpha=0.3, color='#A23B72')
    axes[1].set_xlabel('Budget Limit (€)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Total Satisfaction Score', fontsize=13, fontweight='bold')
    axes[1].set_title('Impact of Budget Constraint', fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: sensitivity_analysis.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("BARATHON OPTIMIZATION - Operations Research Project")
    print(" Authors: Ruben Salaün, Romain Ruan, Jules Nguyen")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate data
    print("Generating Barcelona bar data...")
    venues, distances = generate_realistic_barcelona_data(n_venues=15)  
    print(f"Generated {len(venues)-1} bars + starting point\n")
    
    # Display venue information
    print("Available Venues:")
    print("-"*70)
    for i, v in venues.items():
        if i != 0:
            print(f"  {i}. {v.name:25s} | {v.category:15s} | "
                  f"€{v.cost:5.1f} | Satisfaction: {v.satisfaction}/10 | "
                  f"Stay: {v.stay_time*60:.0f}min")
    print()
    
    # Define constraints
    MAX_TIME = 8.0  # hours - Plus de temps
    MAX_BUDGET = 120.0  # euros - Plus de budget
    
    print(f"Optimization Constraints:")
    print(f"   - Maximum Time Available: {MAX_TIME} hours ({MAX_TIME*60:.0f} minutes)")
    print(f"   - Maximum Budget: €{MAX_BUDGET}")
    print()
    
    # Solve with different algorithms
    solutions = {}
    
    print("Running optimization algorithms...\n")
    
    print("Greedy Algorithm (Fast approximation)...")
    solutions['Greedy'] = solve_greedy_routing_aware(venues, distances, MAX_TIME, MAX_BUDGET)
    print(f"Found solution with satisfaction: {solutions['Greedy']['objective']:.1f}")
    
    print("\n2-Opt Local Search (Improvement heuristic)...")
    solutions['2-Opt'] = solve_two_opt_local_search(venues, distances, MAX_TIME, MAX_BUDGET)
    print(f"Found solution with satisfaction: {solutions['2-Opt']['objective']:.1f}")
    
    # Display all solutions
    for algo_name, sol in solutions.items():
        print_solution_details(sol, venues, distances)
    
    # Compare algorithms
    print(f"\n{'='*70}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Satisfaction':>15} {'Venues':>10} {'Time (h)':>12} {'Cost (€)':>12}")
    print("-"*70)
    for algo_name, sol in solutions.items():
        print(f"{algo_name:<15} {sol['objective']:>15.1f} {sol['n_venues']:>10} "
              f"{sol['total_time']:>12.2f} {sol['total_cost']:>12.2f}")
    print("="*70 + "\n")
    
    # Sensitivity analysis
    sensitivity_df = run_sensitivity_analysis(venues, distances)
    print("Analysis complete\n")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(venues, distances, solutions, sensitivity_df)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\n Generated files:")
    print("  - barathon_route.png - Visual map of optimal route")
    print("  - sensitivity_analysis.png - Parameter sensitivity charts")
    print("\n Interpretation:")
    best_algo = max(solutions.keys(), key=lambda k: solutions[k]['objective'])
    best_sol = solutions[best_algo]
    print(f"  - Best algorithm: {best_algo}")
    print(f"  - You can visit {best_sol['n_venues']} bars")
    print(f"  - Total satisfaction score: {best_sol['objective']:.1f}/10")
    print(f"  - Time efficiency: {best_sol['objective']/best_sol['total_time']:.2f} satisfaction per hour")
    print(f"  - Cost efficiency: {best_sol['objective']/best_sol['total_cost']:.2f} satisfaction per euro")
    print("\n")

if __name__ == "__main__":
    main()

