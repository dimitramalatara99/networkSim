import time
import numpy as np
import matplotlib.pyplot as plt

# modeling space for the base stations and the UEs
array_size = 100
num_timesteps = 50
# step_size = 0.02
packet_size = 2048  # bits
bandwidth = [100_000, 100_000, 100_000]
traffics = []

# coordinates for 3 stationary base stations
bs_positions = np.array([[50, 80], [25, 30], [75, 30]])

# random coordinates for 10 UEs
num_ues = 10


# ue_positions = np.random.randint(0, array_size, size=(num_ues, 2))
def initialize_ues(num_ues, array_size):
    return np.random.randint(1, array_size - 1, size=(num_ues, 2))


# def signal_strength(bs_positions, ue_positions):
#     distance = np.linalg.norm(bs_positions - ue_positions)
#     return 1 / (1 + distance ** 2)

def choose_connection(bs_positions, ue_positions):
    connections = []
    for ue_pos in ue_positions:
        # calculate strength from each base station
        signals = []
        for bs in bs_positions:
            distance = np.linalg.norm(bs - ue_pos)
            signals.append(1 / (1 + distance ** 2))
        # signals = [signal_strength(bs, ue_pos) for bs in bs_positions]
        # best_bs = np.argmax(signals)  # find the best signal
        connections.append(np.argmax(signals))
    return connections


#UEs movement randomly by X step

def move_ues(ue_positions, array_size):
    for i in range(len(ue_positions)):
        # ue can move 1 step in 4 directions(up,right,down,left)
        ue_positions[i] += np.random.randint(-10, 10, size=2)
        # ues move only inside the aray
        ue_positions[i] = np.clip(ue_positions[i], 1, array_size - 1)
    return ue_positions


def data_traffic(num_ues, packet_size):
    # poisson distribution
    packets_per_ue = np.random.poisson(5, num_ues)
    traffics.append(packets_per_ue*packet_size)
    return packets_per_ue * packet_size


def allocate_bandwidth(bs_positions, ue_positions, connections, bandwidth):
    # count how many ues are connected to a BS in a single timestep
    ue_count = [0] * len(bs_positions)
    for connection in connections:
        ue_count[connection] += 1

    ue_bandwith = []  # how much bandwidth is allocated to each ue
    for i, bs_index in enumerate(connections):
        if ue_count[bs_index] > 0:
            bandwidth_per_ue = bandwidth[bs_index] / ue_count[bs_index]
        else:
            bandwidth_per_ue = 0  # this bs has no connections to ues
        ue_bandwith.append(bandwidth_per_ue)
    return ue_bandwith, ue_count


# PLOT
def simulation(bs_positions, ue_positions, connections, traffic, timestep, plt):
    # Plot BS positions in purple
    plt.rcParams["figure.figsize"] = [8, 8]
    plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='purple', label='Base Station')

    for i, (x, y) in enumerate(bs_positions):
        # offset to the right
        plt.text(x + 1, y, f"BS-{i}", color='green', fontsize=9)
    # Plot UEs positions in blue
    plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='blue', label='User Equipment')
    for i, (x, y) in enumerate(ue_positions):
        plt.text(x + 1, y, f"UE-{i}", color='black', fontsize=9)
    # connecting lines
    for i, ue_pos in enumerate(ue_positions):
        connected_bs = bs_positions[connections[i]]
        # plot a line connecting a bs to ue
        plt.plot([ue_pos[0], connected_bs[0]], [ue_pos[1], connected_bs[1]])

    plt.xlim(0, array_size)
    plt.ylim(0, array_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(f"Simulation (Time = {timestep})")

    plt.draw()
    plt.pause(0.3)
    plt.clf()


# simulate movement and traffic generation
def simulate_movement(bs_positions, ue_positions, num_timesteps, array_size, packet_size):
    plt.figure(figsize=(8, 8))

    for timestep in range(num_timesteps):
        print(f"Timestep {timestep + 1}/{num_timesteps}")

        # if timestep % 2 == 0:
        ue_positions = move_ues(ue_positions, array_size)

        connections = choose_connection(bs_positions, ue_positions)

        traffic = data_traffic(num_ues, packet_size)
        print(f"Traffic data (bits) at timestep {timestep + 1}: {traffic}")

        ue_bandwidth, ue_count = allocate_bandwidth(bs_positions, ue_positions, connections, bandwidth)
        print (f"Bandwidth(bits/sec) of ues at timestep{timestep + 1}: {ue_bandwidth}")

        # call simulation function for each new timestep
        simulation(bs_positions, ue_positions, connections, traffic, timestep + 1, plt)



# def simulate(bs_positions, ue_positions):
#     connections = choose_connection(bs_positions, ue_positions)
#     simulation(bs_positions, ue_positions, connections)
# simulate(bs_positions, ue_positions)

if __name__ == '__main__':
    ue_positions = initialize_ues(num_ues, array_size)
    #connections = choose_connection(bs_positions, ue_positions)
    simulate_movement(bs_positions, ue_positions, num_timesteps, array_size, packet_size)

    print("bs positions: ", bs_positions)
    print("ue positions: ", ue_positions)
    print("Best connection for each UE:", connections)
