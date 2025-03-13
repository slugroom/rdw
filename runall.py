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

# Divide the summed probabilities by 3
result["win_probability"] /= 3

# Sort the results by win_probability in descending order
result = result.sort_values(by="win_probability", ascending=False)

# Print the sorted results
print("\n" * 5)
print(result)

new_value0 = 0
new_value1 = 0

first_prob = result.iloc[0]["win_probability"]
second_prob = result.iloc[1]["win_probability"]
third_prob = result.iloc[2]["win_probability"]
new_value0 = (first_prob * 3 + second_prob * 2) / 3
print("\nCalculated Value:", new_value0)

new_value1 = (first_prob * 3 + second_prob * 2 + third_prob) / 3
print("\nCalculated Value:", new_value1)
    
    
total_credits = 590
percent_bet = 100
amount_bet = total_credits * percent_bet / 100
if (new_value0 > new_value1):
    print("Bet top 2")
    total_prob = first_prob + second_prob
    print("Bet 1st: ", amount_bet * first_prob / total_prob)
    print("Bet 2st: ", amount_bet * second_prob / total_prob)
else:
    print("Bet top 3")
    total_prob = first_prob + second_prob + third_prob
    print("Bet 1st: ", amount_bet * first_prob / total_prob)
    print("Bet 2st: ", amount_bet * second_prob / total_prob)
    print("Bet 3rd: ", amount_bet * third_prob / total_prob)