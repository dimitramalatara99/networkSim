import numpy as np
import matplotlib.pyplot as plt
import main
import er3

packet_size = 8192  # bits
bandwidth_per_bs = [50_000, 50_000, 50_000]  # bits/sec
num_timesteps = 100
num_ues = 10

ue_throughput = [0] * num_ues  # Throughput for each UE (bits/sec)
ue_latency = [[] for _ in range(num_ues)]  # Latency/t for each UE
ue_throughput_time = [[] for _ in range(num_ues)]  # Throughput/t
packet_loss_time = [[] for _ in range(num_ues)]  # packet loss/t
bs_bandwidth_using = [[] for _ in range(len(bandwidth_per_bs))]  # Bandwidth usage for each BS


def simulate_with_metrics():
    env = er3.simpy.Environment()
    ue_positions = main.initialize_ues(num_ues, main.array_size)
    bs_positions = main.bs_positions

    def ue_process_with_metrics(env, ue_id, ue_positions, bandwidth_resources):
        while True:
            # Move UE
            main.move_ues([ue_positions[ue_id]], main.array_size)

            # Determine connections
            connections = main.choose_connection(bs_positions, ue_positions)
            connected_bs = connections[ue_id]
            traffic = main.data_traffic(1, packet_size)[0]

            with bandwidth_resources[connected_bs].request() as req:
                yield req

                # Calculate bandwidth per UE
                ue_count = sum(1 for conn in connections if conn == connected_bs)
                bandwidth_per_ue = bandwidth_per_bs[connected_bs] / ue_count if ue_count > 0 else 0

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

            # Track bandwidth usage for the connected BS
            total_bandwidth_used = traffic * ue_count #calculate by traffic generetated from ues connected to the same BS

            # Ensure one value is appended per timestep
            if len(bs_bandwidth_using[connected_bs]) < env.now + 1:
                bs_bandwidth_using[connected_bs].extend([0] * (env.now - len(bs_bandwidth_using[connected_bs]) + 1))
            bs_bandwidth_using[connected_bs][env.now - 1] = total_bandwidth_used / bandwidth_per_bs[connected_bs]

            # Wait for the next timestep
            yield env.timeout(1)

    # Create bandwidth resources
    bandwidth_resources = er3.create_bandwidth_resources(env)

    # Create UE processes
    for ue_id in range(num_ues):
        env.process(ue_process_with_metrics(env, ue_id, ue_positions, bandwidth_resources))

    # Run the simulation
    env.run(until=num_timesteps)

    for bs_id in range(len(bandwidth_per_bs)):
        while len(bs_bandwidth_using[bs_id]) < num_timesteps:
            bs_bandwidth_using[bs_id].append(0)  # Fill missing timesteps with 0

    # Plot results
    plot_all()


def plot_all():
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

    # Plot Bandwidth used
    plt.figure(figsize=(10, 6))
    for bs_id in range(len(bandwidth_per_bs)):
        plt.plot(range(num_timesteps), bs_bandwidth_using[bs_id], label=f'BS-{bs_id}')
    plt.title("Bandwidth Utilization/Time")
    plt.xlabel("Timestep")
    plt.ylabel("Utilization (fraction)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulate_with_metrics()
