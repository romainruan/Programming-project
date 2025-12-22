import math

def generate_data():
    # Fixed venues (coordinates, stay duration, cost, satisfaction)
    venues = {
        0: {"x": 0, "y": 0, "stay": 1.0, "cost": 6, "satisfaction": 4.0},
        1: {"x": 2, "y": 1, "stay": 1.2, "cost": 7, "satisfaction": 4.5},
        2: {"x": 3, "y": 4, "stay": 0.8, "cost": 5, "satisfaction": 3.8},
        3: {"x": 6, "y": 1, "stay": 1.5, "cost": 8, "satisfaction": 4.8},
        4: {"x": 7, "y": 5, "stay": 1.0, "cost": 6, "satisfaction": 4.2},
        5: {"x": 9, "y": 2, "stay": 1.3, "cost": 9, "satisfaction": 5.0},
    }

    # Euclidean distances between venues (travel time)
    distances = {}
    for i in venues:
        for j in venues:
            if i != j:
                dx = venues[i]["x"] - venues[j]["x"]
                dy = venues[i]["y"] - venues[j]["y"]
                distances[(i, j)] = round(math.hypot(dx, dy), 2)

    return venues, distances

