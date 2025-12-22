import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.synthetic_data import generate_data
from models.ilp_model import solve_ilp
from models.baseline import baseline_solution
from experiments.plot_path import plot_path

def main():
    venues, distances = generate_data()

    print("=== Venues ===")
    for i,v in venues.items():
        print(f"Venue {i}: Coord=({v['x']},{v['y']}), Stay={v['stay']}h, Cost={v['cost']}, Satisfaction={v['satisfaction']}")

    # Résolution ILP
    model, x_vars, y_vars, c_vars = solve_ilp(venues, distances, T=15, B=50, return_vars=True)

    print("\n=== Résultat ILP ===")
    print("Status:", model.status)
    print("Objectif (total satisfaction):", round(model.objective.value(),2))

    # Venues sélectionnées
    selected = [i for i in venues if x_vars[i].value() == 1]
    print("\nVenues visited:")
    for i in selected:
        print(f"- Venue {i}, Consumed: {c_vars[i].value():.0f}, Satisfaction: {venues[i]['satisfaction']}")

    # Chemin optimal
    path = [(i,j) for (i,j), var in y_vars.items() if var.value() == 1]
    print("\nEdges travelled (i -> j):")
    if path:
        for e in path:
            print(f"{e[0]} -> {e[1]}")
    else:
        print("No edges selected")

    # Baseline
    baseline_score = baseline_solution(venues)
    print("\n=== Baseline ===")
    print("Total satisfaction (baseline heuristic):", round(baseline_score,2))

    # Sensitivity analysis
    print("\n=== Sensitivity Analysis (Time) ===")
    for T in [10,15,20]:
        model_sa, _, _, _ = solve_ilp(venues, distances, T=T, B=50, return_vars=True)
        print(f"Total time T={T}h -> Objective={round(model_sa.objective.value(),2)}")

    # Visualiser le chemin ILP
    plot_path(venues, path, title="ILP Optimal Venue Path")
    
if __name__ == "__main__":
    main()
