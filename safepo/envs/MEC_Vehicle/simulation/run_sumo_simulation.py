import traci
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_closest_center(x, y, centers):
    return min(centers, key=lambda center: calculate_distance(x, y, centers[center][0], centers[center][1]))


def run_simulation(sumo_binary, config_file):
    traci.start([sumo_binary, "-c", config_file])

    # Define centers of 5x5 grid, each grid cell is 400x400, so centers are at 200, 600, 1000, ...
    # Assigning IDs to each center
    centers = {f"agent_{i*5+j}": (75 + i * 150, 75 + j * 150) for i in range(5) for j in range(5)}
    center_vehicle_count = {center: 0 for center in centers}
    print(centers)
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # Reset vehicle count for each center
            center_vehicle_count = {center: 0 for center in centers}

            for vehicle_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vehicle_id)
                closest_center_id = get_closest_center(x, y, centers)
                center_vehicle_count[closest_center_id] += 1

            # Print vehicle count per center every second
            if step % 1 == 0:  # Adjust if needed for different time intervals
                print(f"Time {step}: {center_vehicle_count}")

            step += 1
    finally:
        traci.close()

run_simulation("sumo", "grid.sumocfg")
