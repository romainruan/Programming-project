import matplotlib.pyplot as plt

def plot_path(venues, edges, title="Optimal Venue Path"):
    """
    Visualize venues and the optimal path
    venues : dictionary containing x, y coordinates
    edges  : list of tuples (i, j)
    """
    fig, ax = plt.subplots()
    
    # Tracer les venues
    for i,v in venues.items():
        ax.scatter(v['x'], v['y'], s=100, c='blue')
        ax.text(v['x']+0.1, v['y']+0.1, f"{i} ({v['satisfaction']})")

    # Tracer les edges
    for i,j in edges:
        x_vals = [venues[i]['x'], venues[j]['x']]
        y_vals = [venues[i]['y'], venues[j]['y']]
        ax.plot(x_vals, y_vals, 'r--', linewidth=1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    plt.grid(True)
    plt.show()

