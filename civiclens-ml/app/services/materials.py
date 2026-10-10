"""Expert-system lookup: (issue type, severity) -> materials, tools and crew."""
from app.constants import DEPARTMENTS


def get_resolution_plan(cls: str, severity: int) -> dict:
    sev = max(1, min(5, int(severity)))
    department = DEPARTMENTS.get(cls, "General Municipal Services")
    materials, tools, crew = [], [], 2

    if cls == "pothole":
        tons = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.5, 5: 1.0}[sev]
        materials = [f"Cold-mix asphalt {tons} t", "Bitumen tack coat 5 L"]
        tools = ["Tamper / plate compactor", "Shovels", "Safety cones"]
        crew = 2 if sev <= 3 else 3
        if sev == 5:
            tools += ["Barricades", "Reflective signage"]
    elif cls == "garbage":
        trips = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}[sev]
        materials = ["Heavy-duty waste bags"]
        tools = ["Garbage truck / tipper", "Gloves and shovels", f"Planned trips: {trips}"]
        crew = 2 if sev <= 2 else 4
    elif cls == "damaged_streetlight":
        units = 1 if sev <= 3 else 2
        materials = [f"Replacement LED lamp unit x{units}", "Wiring kit", "Fuse set"]
        tools = ["Aerial work platform / ladder truck", "Insulated tools", "Multimeter"]
        crew = 2
        if sev >= 4:
            tools.append("Electrical safety cordon (possible exposed wiring)")
    elif cls == "waterlogging":
        pumps = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}[sev]
        materials = ["Drain-clearing consumables"]
        tools = ["Jetting / desilting machine", "Manhole keys", "Safety barricades"]
        if pumps:
            tools.append(f"Portable dewatering pumps x{pumps}")
        crew = 3 if sev <= 3 else 5
    else:
        materials = ["To be assessed by field officer"]
        tools = ["Inspection kit"]

    return {"department": department, "materials": materials, "tools": tools, "crew_size": crew}
