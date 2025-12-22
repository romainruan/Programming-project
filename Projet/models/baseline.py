def baseline_solution(venues, distances=None, T=15, B=50):
    """
    Baseline heuristic:
    - Greedy on satisfaction
    - Respect time T and budget B constraints
    """
    selected = []
    total_time = 0
    total_cost = 0
    total_satisfaction = 0

    # Sort venues by decreasing satisfaction
    for i, v in sorted(venues.items(), key=lambda item: item[1]['satisfaction'], reverse=True):
        if total_time + v['stay'] <= T and total_cost + v['cost'] <= B:
            selected.append(i)
            total_time += v['stay']
            total_cost += v['cost']
            total_satisfaction += v['satisfaction']

    return total_satisfaction

