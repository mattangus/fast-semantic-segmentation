
def reverse(d):
    return {b: a for a,b in d.items()}

name_rgb = {"bridge": (0,0,255),
            "building": (0,255,0),
            "construction_barrel": (0,255,255),
            "construction_barricade": (255,0,0),
            "crosswalk": (255,0,255),
            "curb": (255,255,0),
            "sky": (255,255,255),
            "debris": (41,41,41),
            "fence": (156,41,41),
            "guard_rail": (99,99,41),
            "lane_separator": (214,99,41),
            "pavement_marking": (41,156,41),
            "rail_track": (156,156,41),
            "road": (99,214,41),
            "roadside_structure": (214,214,41),
            "rumble_strip": (99,41,99),
            "sidewalk": (214,41,99),
            "terrain": (41,99,99),
            "traffic_cone": (156,99,99),
            "traffic_light": (99,156,99),
            "traffic_marker": (214,156,99),
            "traffic_sign": (41,214,99),
            "tunnel": (156,214,99),
            "utility_pole": (41,41,156),
            "vegetation": (156,41,156),
            "wall": (99,99,156),
            "Car": (214,99,156),
            "Trailer": (41,156,156),
            "Bus": (156,156,156),
            "Truck": (99,214,156),
            "Airplane": (214,214,156),
            "Moterbike": (99,41,214),
            "Bycicle": (41,99,214),
            "Boat": (156,99,214),
            "Railed": (99,156,214),
            "Pedestrian": (214,156,214),
            "Animal": (156,214,214)}

rgb_name = reverse(name_rgb)

rgb_train = {(0, 0, 255): 0,
            (0, 255, 0): 1,
            (255, 255, 255): 2,
            (214, 156, 214): 3,
            (255, 0, 255): 4,
            (255, 255, 0): 5,
            (41, 41, 41): 6,
            (156, 41, 41): 7,
            (99, 99, 41): 8,
            (214, 99, 41): 9,
            (41, 156, 41): 10,
            (156, 156, 41): 11,
            (99, 214, 41): 12,
            (99, 41, 99): 13,
            (214, 41, 99): 13,
            (41, 99, 99): 14,
            (156, 99, 99): 15,
            (99, 214, 156): 16,
            (99, 156, 99): 17,
            (41, 214, 99): 18,
            (156, 214, 99): 19,
            (214, 214, 41): 20,
            (156, 156, 156): 21,
            (156, 41, 156): 22,
            (214, 99, 156): 23,
            (99, 99, 156): 24,
            (41, 41, 156): 25}

train_rgb = reverse(rgb_train)

name_train = {a: rgb_train[b] for a, b in name_rgb.items() if b in rgb_train}
train_name = reverse(name_train)

print(train_name)