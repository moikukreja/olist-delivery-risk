"""
07_simplify_geojson.py
----------------------
Shrink the Brazil state-boundary map file so it can be shipped inside the app.

THE PROBLEM
-----------
The public GeoJSON of Brazil's 27 states is 5.5 MB. That is a huge download for
a web page - it would take seconds to load and would dominate the app's size.

THE FIX
-------
Two lossless-looking tricks:

  1. DOUGLAS-PEUCKER SIMPLIFICATION
     A coastline drawn with 10,000 points looks identical to one drawn with 400
     when displayed on a screen a few hundred pixels wide. This algorithm walks
     the outline and throws away any point that sits almost exactly on the
     straight line between its neighbours.

  2. COORDINATE ROUNDING
     The raw file stores longitude as -46.633308837890625. At the zoom level we
     display, -46.633 is indistinguishable. Cutting to 3 decimal places (about
     100 m of precision) removes a dozen characters from every single number.

Run it with:
    venv\\Scripts\\python.exe src\\07_simplify_geojson.py
"""

import json
import sys

from utils import PROJECT_ROOT, banner, sub_banner

# Bigger tolerance = smaller file but blockier outlines. 0.02 degrees is about
# 2 km, which is invisible on a map of a country 4,000 km wide.
TOLERANCE = 0.02
DECIMALS = 3

RAW = PROJECT_ROOT / "frontend" / "br_states_raw.json"
OUT = PROJECT_ROOT / "frontend" / "src" / "assets" / "brazil-states.json"


def perpendicular_distance(point, start, end) -> float:
    """How far is `point` from the straight line joining `start` and `end`?"""
    (px, py), (sx, sy), (ex, ey) = point, start, end
    dx, dy = ex - sx, ey - sy
    if dx == 0 and dy == 0:
        return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
    # Standard point-to-line-segment distance formula.
    return abs(dy * px - dx * py + ex * sy - ey * sx) / ((dx * dx + dy * dy) ** 0.5)


def douglas_peucker(points: list, tolerance: float) -> list:
    """Keep only the points that actually change the shape of the outline."""
    if len(points) < 3:
        return points

    # Find the point that strays furthest from the straight line end-to-end.
    furthest_index, furthest_distance = 0, 0.0
    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], points[0], points[-1])
        if distance > furthest_distance:
            furthest_index, furthest_distance = i, distance

    # If even the worst point is close to the line, the whole run can become a
    # single straight segment. Otherwise split at that point and repeat.
    if furthest_distance <= tolerance:
        return [points[0], points[-1]]

    left = douglas_peucker(points[: furthest_index + 1], tolerance)
    right = douglas_peucker(points[furthest_index:], tolerance)
    return left[:-1] + right


def simplify_ring(ring: list) -> list | None:
    """Simplify one closed outline, then round its numbers."""
    simplified = douglas_peucker([tuple(p[:2]) for p in ring], TOLERANCE)
    if len(simplified) < 4:          # too few points to form a shape - drop it
        return None
    rounded = [[round(x, DECIMALS), round(y, DECIMALS)] for x, y in simplified]
    if rounded[0] != rounded[-1]:    # a polygon must end where it started
        rounded.append(rounded[0])
    return rounded


def simplify_geometry(geometry: dict) -> dict | None:
    kind = geometry["type"]

    if kind == "Polygon":
        rings = [simplify_ring(r) for r in geometry["coordinates"]]
        rings = [r for r in rings if r]
        return {"type": "Polygon", "coordinates": rings} if rings else None

    if kind == "MultiPolygon":
        polygons = []
        for polygon in geometry["coordinates"]:
            rings = [simplify_ring(r) for r in polygon]
            rings = [r for r in rings if r]
            if rings:
                polygons.append(rings)
        return {"type": "MultiPolygon", "coordinates": polygons} if polygons else None

    return None


def count_points(geometry: dict) -> int:
    if geometry["type"] == "Polygon":
        return sum(len(r) for r in geometry["coordinates"])
    return sum(len(r) for p in geometry["coordinates"] for r in p)


def main() -> int:
    banner("SIMPLIFYING THE BRAZIL MAP FILE")

    if not RAW.exists():
        print(f"  ERROR: {RAW} not found. Download the raw GeoJSON first.")
        return 1

    # utf-8-sig strips the invisible "byte order mark" some editors add.
    raw = json.loads(RAW.read_text(encoding="utf-8-sig"))
    print(f"  Input : {RAW.name}  ({RAW.stat().st_size / 1024**2:.2f} MB)")
    print(f"  States: {len(raw['features'])}")
    print(f"  Sample properties: {json.dumps(raw['features'][0]['properties'])[:120]}")

    sub_banner("Simplifying each state outline")
    features, points_before, points_after = [], 0, 0

    for feature in raw["features"]:
        properties = feature.get("properties", {})
        # Different sources name the state-code field differently, so we try
        # every likely spelling and keep the first two-letter code we find.
        code = None
        for key in ("SIGLA", "sigla", "UF", "uf", "abbrev_state", "postal", "SIGLA_UF"):
            value = properties.get(key)
            if isinstance(value, str) and len(value) == 2:
                code = value.upper()
                break
        name = properties.get("NOME") or properties.get("name") or properties.get("nome") or code

        points_before += count_points(feature["geometry"])
        geometry = simplify_geometry(feature["geometry"])
        if geometry is None:
            print(f"    skipped {code or name} - geometry collapsed")
            continue
        points_after += count_points(geometry)

        features.append({
            "type": "Feature",
            "properties": {"code": code, "name": name},
            "geometry": geometry,
        })

    output = {"type": "FeatureCollection", "features": features}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    banner("RESULT")
    print(f"  States kept   : {len(features)}")
    print(f"  Outline points: {points_before:,}  ->  {points_after:,}  "
          f"({(1 - points_after / points_before) * 100:.1f}% removed)")
    print(f"  File size     : {RAW.stat().st_size / 1024**2:.2f} MB  ->  {size_kb:.0f} KB "
          f"({(1 - size_kb * 1024 / RAW.stat().st_size) * 100:.1f}% smaller)")
    print(f"  Saved         : {OUT}")
    codes = sorted(f["properties"]["code"] for f in features if f["properties"]["code"])
    print(f"  State codes   : {', '.join(codes)}")
    print()
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(20000)
    sys.exit(main())
