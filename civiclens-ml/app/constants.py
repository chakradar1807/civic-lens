"""Issue classes, department routing and base severity. Edit here to add classes."""

CLASSES = ["pothole", "garbage", "damaged_streetlight", "waterlogging"]

# Names that public datasets / Roboflow exports commonly use -> our canonical class names
ALIASES = {
    "potholes": "pothole", "pot_hole": "pothole", "d40": "pothole", "road_damage": "pothole",
    "trash": "garbage", "litter": "garbage", "waste": "garbage", "garbage_dump": "garbage",
    "streetlight": "damaged_streetlight", "street_light": "damaged_streetlight",
    "broken_streetlight": "damaged_streetlight", "broken_street_light": "damaged_streetlight",
    "dark_streetlight": "damaged_streetlight", "lamp_post": "damaged_streetlight",
    "flood": "waterlogging", "flooded_street": "waterlogging", "water_logging": "waterlogging",
    "flooding": "waterlogging", "waterlogged": "waterlogging",
}

DEPARTMENTS = {
    "pothole": "Roads & Highways",
    "garbage": "Solid Waste Management",
    "damaged_streetlight": "Electricity & Street Lighting",
    "waterlogging": "Storm Water Drainage",
}

# Starting severity of each issue type before size/context adjustments (1-5)
BASE_SEVERITY = {
    "pothole": 2,
    "garbage": 1,
    "damaged_streetlight": 2,
    "waterlogging": 3,
}

SEVERITY_LABELS = {1: "Minor", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}

DISPATCH_POLICY = {
    "CRITICAL": "Dispatch emergency crew immediately (target: within 15 minutes)",
    "HIGH": "Dispatch crew in the next available slot (target: within 2 hours)",
    "MEDIUM": "Schedule within 24 hours",
    "LOW": "Schedule within 7 days (batch with nearby tickets)",
}


def normalize_class(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, key)
