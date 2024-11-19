import requests
import csv
import os
from datetime import datetime

# API endpoint and parameters
api_url = "https://bustime.mta.info/api/siri/vehicle-monitoring.json"
api_key = "730d003a-a21d-431a-b261-f8a74ee9c06d"  
line_ref = "S62"

# Fetch data from the MTA API
params = {
    "key": api_key,
    "OperatorRef": "MTA NYCT",
    "LineRef": line_ref
}

response = requests.get(api_url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # Navigate through the JSON structure to extract bus data
    vehicle_activities = data.get("Siri", {}).get("ServiceDelivery", {}).get("VehicleMonitoringDelivery", [])[0].get("VehicleActivity", [])

    # Prepare the CSV file
    csv_filename = "bus_data.csv"
    is_empty = not os.path.isfile(csv_filename) or os.stat(csv_filename).st_size == 0  # Check if the file is empty

    with open(csv_filename, mode='a', newline='') as file:  # Open in append mode
        writer = csv.writer(file)
        
        # Write the header row only if the file is empty
        if is_empty:
            writer.writerow(["Date", "Day", "Route", "Direction", "Bus ID", "Stop Name", "Arrival Time", "Passenger Count"])
        
        # Write the data rows
        for activity in vehicle_activities:
            journey = activity.get("MonitoredVehicleJourney", {})
            monitored_call = journey.get("MonitoredCall", {})
            capacities = monitored_call.get("Extensions", {}).get("Capacities", {})
            
            # Extract required fields
            date = datetime.now().strftime("%Y-%m-%d")  # Current date
            day = datetime.now().strftime("%A")  # Current day
            route = journey.get("PublishedLineName", "N/A")
            direction = journey.get("DirectionRef", "N/A")
            bus_id = journey.get("VehicleRef", "N/A")
            stop_name = monitored_call.get("StopPointName", "N/A")
            arrival_time = monitored_call.get("ExpectedArrivalTime", "N/A")
            passenger_count = capacities.get("EstimatedPassengerCount", "N/A")
            
            # Write the row to the CSV file
            writer.writerow([date, day, route, direction, bus_id, stop_name, arrival_time, passenger_count])

    print(f"Data appended to {csv_filename}")
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

