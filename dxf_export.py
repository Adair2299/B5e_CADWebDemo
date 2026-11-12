
# Export CAD executor geometry to an editable DXF file.

import math
import ezdxf

COLORS = {"red": 1, "yellow": 2, "green": 3, "cyan": 4, "blue": 5, "magenta": 6, "white": 7, "black": 7}

def export_dxf(objects, filename):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for obj in objects.values():
        kind = obj["type"]
        color = COLORS.get(obj.get("color", "black"), 7)
        attrs = {"color": color}

        if kind == "line":
            msp.add_line((obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), dxfattribs=attrs)

        elif kind == "circle":
            msp.add_circle((obj["x"], obj["y"]), obj["radius"], dxfattribs=attrs)

        elif kind == "point":
            msp.add_point((obj["x"], obj["y"]), dxfattribs=attrs)

        elif kind == "rectangle":
            x, y = obj["x"], obj["y"]
            w, h = obj["geometry_width"], obj["height"]
            angle = math.radians(obj.get("angle", 0))
            corners = [(0, 0), (w, 0), (w, h), (0, h)]
            points = [(x + a * math.cos(angle) - b * math.sin(angle),
                       y + a * math.sin(angle) + b * math.cos(angle)) for a, b in corners]
            msp.add_lwpolyline(points, close=True, dxfattribs=attrs)

    doc.saveas(filename)
