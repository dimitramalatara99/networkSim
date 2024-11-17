import simpy
import numpy as np
import matplotlib.pyplot as plt
import main

packet_size = 2048  # bits(packet size = 256 bytes)
bandwidth_per_bs = [100_000, 100_000, 100_000]  # bits/sec
num_timesteps = 100
num_ues = 10

bs_positions = main.bs_positions

ue_throughput = [0] * num_ues  # Throughput (bits/s) for each UE
ue_data = [0] * num_ues  # Total data transmitted (bits) for each UE
ue_latency = [[] for _ in range(num_ues)]  # Latency recorded at each timestep
ue_throughput_over_time = [[] for _ in range(num_ues)]  # Throughput over time
traffics = []

def ue_process(env, ue_id, ue_positions, bandwidth_resources):
    while True:
        main.move_ues([ue_positions[ue_id]], main.array_size)

        connections = main.choose_connection(bs_positions, ue_positions)
        connected_bs = connections[ue_id]
        traffic = main.data_traffic(1, packet_size)[0]


        with bandwidth_resources[connected_bs].request() as req:
            yield req
            # Calculate latency based on allocated bandwidth
            bandwidth_per_ue = bandwidth_per_bs[connected_bs] / sum(1 for conn in connections if conn == connected_bs)
            latency = packet_size / bandwidth_per_ue

            # Update total data transmitted
            ue_data[ue_id] += traffic

            # Calculate throughput
            time_elapsed = env.now if env.now > 0 else 1  # Avoid division by zero
            ue_throughput[ue_id] = ue_data[ue_id] / time_elapsed

            # Record latency and throughput
            ue_latency[ue_id].append(latency)
            ue_throughput_over_time[ue_id].append(ue_throughput[ue_id])

            # Log the event
            print(f"Time {env.now}: UE-{ue_id} sent {traffic} bits to BS-{connected_bs} "
                  f"(Latency: {latency:.4f} seconds, Throughput: {ue_throughput[ue_id]:.2f} bits/s)")

        # Wait for the next timestep
        yield env.timeout(1)



def create_bandwidth_resources(env):
    return [simpy.Resource(env, capacity=b) for b in bandwidth_per_bs]


#  simulation
def simulate():
    env = simpy.Environment()

    # Initialize UE positions using shared logic from main.py
    ue_positions = main.initialize_ues(num_ues, main.array_size)

    # Create bandwidth resources
    bandwidth_resources = create_bandwidth_resources(env)

    # Create UE processes
    for ue_id in range(num_ues):
        env.process(ue_process(env, ue_id, ue_positions, bandwidth_resources))


    # Run the simulation for a fixed number of timesteps
    env.run(until=num_timesteps)
    print(f" Traffic per ue: {traffics}")
    # Plot results after the simulation
    plot_metrics(ue_latency, ue_throughput_over_time, num_timesteps)


# Plot latency and throughput over time
def plot_metrics(ue_latency, ue_throughput_over_time, num_timesteps):
    # Plot Latency
    plt.figure(figsize=(10, 6))
    for ue_id in range(len(ue_latency)):
        plt.plot(range(num_timesteps), ue_latency[ue_id], label=f'UE-{ue_id}')
    plt.title("Latency")
    plt.xlabel("Timestep")
    plt.ylabel("Latency (seconds)")
    plt.legend()
    plt.show()

    # Plot Throughput
    plt.figure(figsize=(10, 6))
    for ue_id in range(len(ue_throughput_over_time)):
        plt.plot(range(num_timesteps), ue_throughput_over_time[ue_id], label=f'UE-{ue_id}')
    plt.title("Throughput")
    plt.xlabel("Timestep")
    plt.ylabel("Throughput (bits/second)")
    plt.legend()
    plt.show()


# Main execution
if __name__ == '__main__':
    simulate()
