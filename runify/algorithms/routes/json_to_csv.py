import json
import csv

def json_to_csv(json_string, csv_filename):
    # Parse JSON data
    data = json.loads(json_string)
    # Ensure data is a list
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError('JSON must be an object or list of objects.')
    # Gather all columns
    columns = set()
    for item in data:
        columns.update(item.keys())
    columns = sorted(list(columns))
    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return f'CSV file {csv_filename} created with columns: {columns}'

# Example usage
file = 'activities.json'
with open(file, 'r') as f:
    example_json = f.read()
json_to_csv(example_json, 'activities.csv')
