import pulp

def solve_ilp(venues, distances, T=15, B=50, return_vars=False):
    model = pulp.LpProblem("VenueHopping", pulp.LpMaximize)

    n = len(venues)

    # Variables
    x = pulp.LpVariable.dicts("x", venues.keys(), 0, 1, pulp.LpBinary)          # visit venue i
    c = pulp.LpVariable.dicts("c", venues.keys(), 0, 1)                         # consumption 0 or 1
    y = pulp.LpVariable.dicts("y", distances.keys(), 0, 1, pulp.LpBinary)       # travel i->j

    # Objective: maximize satisfaction
    model += pulp.lpSum(venues[i]["satisfaction"] * c[i] for i in venues)

    # Total time constraints
    model += pulp.lpSum(venues[i]["stay"] * x[i] for i in venues) + \
             pulp.lpSum(distances[e] * y[e] for e in distances) <= T

    # Budget constraints
    model += pulp.lpSum(venues[i]["cost"] * c[i] for i in venues) <= B

    # Consumption ≤ visit
    for i in venues:
        model += c[i] <= x[i]

    # Each visited venue has exactly one entry and one exit (simplified flow constraint)
    for i in venues:
        model += pulp.lpSum(y[(j,i)] for j in venues if j != i) == x[i]
        model += pulp.lpSum(y[(i,j)] for j in venues if j != i) == x[i]

    # Solve
    model.solve()

    if return_vars:
        return model, x, y, c
    else:
        return model

