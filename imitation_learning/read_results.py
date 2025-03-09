import os
import json

directory = "results/auto"

uncommented_params = [
    "maxSpeed",
    "brakingDuringTurningSpeedThresholdParameter",
    "accelerationDuringTurningParameter",
    "brakingDuringTurningParameter",
    "accelerationParameter",
]

results = []

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), "r") as file:
            data = json.load(file)
            mean_reward = data.get("mean_reward", data.get("mean", {}))
            std_reward = data.get("std_reward", data.get("std", {}))
            results.append({
                "filename": filename,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "parameters": data.get("parameters", {})
            })

# sorted_results = sorted(results, key=lambda x: (x["parameters"]["maxSpeed"], x["mean_reward"]), reverse=True)
sorted_results = sorted(results, key=lambda x: (x["mean_reward"], -x["std_reward"]), reverse=True)

num_results = min(15, len(sorted_results))  # Choose how many top results to display
print("Top", num_results, "results:")
for i in range(num_results):
    result = sorted_results[i]
    print("File:", result["filename"])
    print("Mean Reward:", result["mean_reward"])
    print("Standard Deviation:", result["std_reward"])
    print("Parameters:")
    # params = result["parameters"]
    # for param in uncommented_params:
    #     print(param + ":", params[param])
    print()
