import subprocess
import pandas as pd

# Define the list of scripts to run in order
scripts = ["vehicle.py", "test.py", "test2.py", "test3.py"]

# Run each script one after the other
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)

# Once all scripts have executed, process the CSV files
filenames = ["predictions.csv", "predictions2.csv", "predictions3.csv"]

# Read each CSV file into a DataFrame and combine them
dfs = [pd.read_csv(f) for f in filenames]
df = pd.concat(dfs, ignore_index=True)

# Group by driver_id and driver_name, summing the win_probability for each driver
result = df.groupby(["driver_id", "driver_name"], as_index=False)["win_probability"].sum()

# Sort the results by win_probability in descending order
result = result.sort_values(by="win_probability", ascending=False)

# Print the sorted results
print(result)
