import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# modeling space for the base staions and the UEs
array_size = 100

# coordinates for stationary 3 base stations
bs_positions = np.array([[20, 20], [50, 50], [80, 80]])


# random coordinates for 10 UEs
num_ues = 10
# ue_positions = np.random.randint(0, array_size, size=(num_ues, 2))
def initialize_ues(num_ues, array_size):
    return np.random.randint(0, array_size, size=(num_ues, 2))

# print("bs positions: ", bs_positions)
# print("ue positions: ", ue_positions)


def signal_strength(bs_positions, ue_positions):
    distance = np.linalg.norm(bs_positions - ue_positions)
    # distance = cdist(bs_positions, ue_positions, 'euclidean')
    return 1 / (1 + distance ** 2)

def choose_connection(bs_positions, ue_positions):
    connections = []
    for ue_pos in ue_positions:
        # calculate strength from each base station
        signals = [signal_strength(bs, ue_pos) for bs in bs_positions]
        best_bs = np.argmax(signals)  # find the best signal
        connections.append(best_bs)
    return connections

#connections= choose_connection(bs_positions, ue_positions)

# print("Best connection for each UE:", connections)

# UEs movement randomly by 1 step

def move_ues(ue_positions, array_size):
    for i in range(len(ue_positions)):
        ue_positions[i] += np.random.randint(-1, 2, size=2)
        # ues move only inside the aray
        ue_positions[i] = np.clip(ue_positions[i], 0, array_size - 1)
    return ue_positions



#SIMULATION
def simulation(bs_positions, ue_positions, connections):
    plt.figure(figsize=(8, 8))
    # BS positions in purple
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='purple', label='Base Station')
    # UEs positions in blue
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', label='UEs')

#connecting lines
    for i, ue_pos in enumerate(ue_positions):
        connected_bs = bs_positions[connections[i]]
        # plot a line connecting a bs to ue
        plt.plot([ue_pos[0], connected_bs[0]], [ue_pos[1], connected_bs[1]], 'k--')

    plt.xlim(0, array_size)
    plt.ylim(0, array_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Simulation")
    plt.show()

# def simulate(bs_positions, ue_positions):
#     connections = choose_connection(bs_positions, ue_positions)
#     simulation(bs_positions, ue_positions, connections)
#
#
# simulate(bs_positions, ue_positions)

if __name__ == '__main__':
    ue_positions = initialize_ues(num_ues, array_size)
    connections = choose_connection(ue_positions, bs_positions)
    simulation(bs_positions, ue_positions, connections)

print("bs positions: ", bs_positions)
print("ue positions: ", ue_positions)
print("Best connection for each UE:", connections)