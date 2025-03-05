import requests

base_url = "https://www.ebi.ac.uk/ipd/mhc/ws/alleles"
query = "element(xref,type,IPD-PDB)"
batch_size = 500
start = 0
all_data = []

while True:
    params = {
        'query': query,
        'start': start,
        'size': batch_size
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if not data:
        break
    all_data.extend(data)
    start += batch_size

# Process all_data as needed