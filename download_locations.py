"""
Download complete India States & Districts JSON and save as static/india_locations.json
Source: https://github.com/sab99r/Indian-States-And-Districts (official districts.nic.in data)
"""
import json
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/sab99r/Indian-States-And-Districts/master/states-and-districts.json"

def main():
    print("Downloading complete India locations...")
    with urllib.request.urlopen(RAW_URL, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    # Convert to simple { "State Name": ["District1", "District2", ...] }
    locations = {}
    for item in data.get("states", []):
        state = item.get("state")
        districts = item.get("districts", [])
        if state and districts:
            locations[state] = districts

    out_dir = Path("static")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "india_locations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, ensure_ascii=False, indent=2)

    total_districts = sum(len(d) for d in locations.values())
    print(f"Saved {len(locations)} states/UTs and {total_districts} districts → {out_path}")

if __name__ == "__main__":
    main()
