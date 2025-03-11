import requests
import pandas as pd

def get_next_race():
    url = "https://rdwvirtualracing.azurewebsites.net/Races/Next"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch next race data: {response.status_code}")

def get_drivers():
    url = "https://rdwvirtualracing.azurewebsites.net/Drivers"
    response = requests.get(url)
    if response.status_code == 200:
        return {driver["id"]: driver for driver in response.json()}
    else:
        raise Exception(f"Failed to fetch drivers data: {response.status_code}")

def get_track_info(track_id, weather, time_of_day):
    url = f"https://rdwvirtualracing.azurewebsites.net/Tracks/{track_id}"
    response = requests.get(url)
    if response.status_code == 200:
        track_data = response.json()
        track_data.update({"weather": weather, "time_of_day": time_of_day})
        return track_data
    else:
        raise Exception(f"Failed to fetch track data for track {track_id}: {response.status_code}")

def get_vehicle_data(license_plate):
    url = f"https://opendata.rdw.nl/resource/m9d7-ebf2.json?kenteken={license_plate}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return {
                "kenteken": license_plate,
                "massa_rijklaar": data[0].get("massa_rijklaar"),
                "cilinderinhoud": data[0].get("cilinderinhoud"),
                "vermogen_massarijklaar": data[0].get("vermogen_massarijklaar")
            }
    else:
        raise Exception(f"Failed to fetch data for vehicle {license_plate}: {response.status_code}")

def get_results():
    url = "https://rdwvirtualracing.azurewebsites.net/Results"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch results data: {response.status_code}")

def main():
    race_data = get_next_race()
    drivers = get_drivers()
    
    weather = race_data.get("weather")
    time_of_day = race_data.get("timeOfDay")
    track_id = race_data.get("trackId")
    track_info = get_track_info(track_id, weather, time_of_day) if track_id else {}
    
    vehicle_data_list = []
    
    for participant in race_data["participants"]:
        vehicle_data = get_vehicle_data(participant["vehicle"])
        driver_data = drivers.get(participant["driverId"], {})
        
        if vehicle_data:
            vehicle_data.update({
                "driver_id": driver_data.get("id"),
                "driver_name": driver_data.get("name"),
                "years_of_experience": driver_data.get("yearsOfExperience"),
                "driving_style": driver_data.get("drivingStyle")
            })
            vehicle_data_list.append(vehicle_data)
    
    vehicle_df = pd.DataFrame(vehicle_data_list)
    track_df = pd.DataFrame([track_info])
    
    # Write vehicles.csv and track.csv
    vehicle_df.to_csv("vehicles.csv", index=False)
    track_df.to_csv("track.csv", index=False)
    
    # Fetch results data and write to results.csv
    results_data = get_results()
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("results.csv", index=False)
    
    return vehicle_df, track_df, results_df

if __name__ == "__main__":
    vehicle_df, track_df, results_df = main()
    print(vehicle_df)
    print("Track Info:", track_df)
    print("Results:", results_df)
