import numpy as np
import matplotlib.pyplot as plt
import main
import er3

packet_size = 8192  # bits
bandwidth_per_bs = [50_000, 50_000, 50_000]  # bits/sec
num_timesteps = 100
num_ues = 10

# Metrics tracking
ue_throughput = [0] * num_ues  # Throughput for each UE (bits/sec)
ue_latency = [[] for _ in range(num_ues)]  # Latency/t for each UE
ue_throughput_time = [[] for _ in range(num_ues)]  # Throughput/t
packet_loss_time = [[] for _ in range(num_ues)]  # Packet loss/t
bs_bandwidth_using = [[] for _ in range(len(bandwidth_per_bs))]  # Bandwidth usage for each BS


# Dynamic Bandwidth Allocation Method
def dynamic_bandwidth_allocation(ue_positions, connections, bs_positions, packet_sizes, bandwidth_per_bs):
    """
    Dynamically allocate bandwidth based on user demands (e.g., packet sizes).
    """
    ue_bandwidth = [0] * len(ue_positions)  # Bandwidth allocated to each UE
    bs_usage = [0] * len(bs_positions)  # Total bandwidth usage per BS

    # Group UEs by connected BS
    bs_to_ues = {i: [] for i in range(len(bs_positions))}
    for ue_id, bs_id in enumerate(connections):
        bs_to_ues[bs_id].append(ue_id)

    # Allocate bandwidth for each BS
    for bs_id, ues in bs_to_ues.items():
        if not ues:
            continue

        # Calculate total demand for this BS
        total_demand = sum(packet_sizes[ue_id] for ue_id in ues)

        for ue_id in ues:
            # Allocate bandwidth proportionally to the user's packet size
            ue_demand = packet_sizes[ue_id]
            ue_bandwidth[ue_id] = (ue_demand / total_demand) * bandwidth_per_bs[bs_id]
            bs_usage[bs_id] += ue_bandwidth[ue_id]

    return ue_bandwidth, bs_usage


def simulate_with_dynamic_bandwidth():
    env = er3.simpy.Environment()
    ue_positions = main.initialize_ues(num_ues, main.array_size)
    bs_positions = main.bs_positions

    def ue_process_with_dynamic_bandwidth(env, ue_id, ue_positions, bandwidth_resources):
        while True:
            # Move UE
            main.move_ues([ue_positions[ue_id]], main.array_size)

            # Determine connections
            connections = main.choose_connection(bs_positions, ue_positions)
            connected_bs = connections[ue_id]
            traffic = main.data_traffic(1, packet_size)[0]

            # Use dynamic bandwidth allocation
            ue_bandwidth, bs_usage = dynamic_bandwidth_allocation(
                ue_positions, connections, bs_positions, [packet_size] * len(ue_positions), bandwidth_per_bs
            )
            bandwidth_per_ue = ue_bandwidth[ue_id]

            # Calculate latency
            latency = packet_size / bandwidth_per_ue if bandwidth_per_ue > 0 else float('inf')

            # Calculate packet loss
            packet_loss = max(0, packet_size - bandwidth_per_ue)

            # Update total data and throughput
            time_elapsed = env.now if env.now > 0 else 1
            ue_throughput[ue_id] = (traffic - packet_loss) / time_elapsed if time_elapsed > 0 else 0

            # Record metrics
            ue_latency[ue_id].append(latency)
            ue_throughput_time[ue_id].append(ue_throughput[ue_id])
            packet_loss_time[ue_id].append(packet_loss)

            # Record BS bandwidth usage
            if len(bs_bandwidth_using[connected_bs]) < env.now + 1:
                bs_bandwidth_using[connected_bs].extend([0] * (env.now - len(bs_bandwidth_using[connected_bs]) + 1))
            bs_bandwidth_using[connected_bs][env.now - 1] = bs_usage[connected_bs] / bandwidth_per_bs[connected_bs]

            # Wait for the next timestep
            yield env.timeout(1)

    # Create bandwidth resources
    bandwidth_resources = er3.create_bandwidth_resources(env)

    # Create UE processes
    for ue_id in range(num_ues):
        env.process(ue_process_with_dynamic_bandwidth(env, ue_id, ue_positions, bandwidth_resources))

    # Run the simulation
    env.run(until=num_timesteps)

    # Fill missing timesteps for BS bandwidth usage
    for bs_id in range(len(bandwidth_per_bs)):
        while len(bs_bandwidth_using[bs_id]) < num_timesteps:
            bs_bandwidth_using[bs_id].append(0)

    # Plot results
    plot_all_metrics()


def plot_all_metrics():
    # Plot Latency
    plt.figure(figsize=(10, 6))
    for ue_id in range(num_ues):
        plt.plot(range(num_timesteps), ue_latency[ue_id], label=f'UE-{ue_id}')
    plt.title("Latency/Time")
    plt.xlabel("Timestep")
    plt.ylabel("Latency (seconds)")
    plt.legend()
    plt.show()

    # Plot Throughput
    plt.figure(figsize=(10, 6))
    for ue_id in range(num_ues):
        plt.plot(range(num_timesteps), ue_throughput_time[ue_id], label=f'UE-{ue_id}')
    plt.title("Throughput/Time")
    plt.xlabel("Timestep")
    plt.ylabel("Throughput (bits/second)")
    plt.legend()
    plt.show()

    # Plot Packet Loss
    plt.figure(figsize=(10, 6))
    for ue_id in range(num_ues):
        plt.plot(range(num_timesteps), packet_loss_time[ue_id], label=f'UE-{ue_id}')
    plt.title("Packet Loss/Time")
    plt.xlabel("Timestep")
    plt.ylabel("Packet Loss (bits)")
    plt.legend()
    plt.show()

    # Plot Bandwidth Utilization
    plt.figure(figsize=(10, 6))
    for bs_id in range(len(bandwidth_per_bs)):
        plt.plot(range(num_timesteps), bs_bandwidth_using[bs_id], label=f'BS-{bs_id}')
    plt.title("Bandwidth Utilization/Time")
    plt.xlabel("Timestep")
    plt.ylabel("Utilization (fraction)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulate_with_dynamic_bandwidth()
