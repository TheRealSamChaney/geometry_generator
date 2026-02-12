import svgwrite
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.errors import GEOSException
try:
    # Optional API (depends on Shapely version).
    from shapely import symmetric_difference_all
except ImportError:
    symmetric_difference_all = None

# This will be a SVG geometric pattern creator app

width = 400
height = 400

output_folder = os.path.dirname(os.path.abspath(__file__))
output_filename = "geometry_generator_output.svg"
output_path = os.path.join(output_folder, output_filename)
drawing_global = svgwrite.Drawing(
    output_path,
    size=(f"{width}px", f"{height}px"),
    viewBox=f"0 0 {width} {height}",
)

class Polygon:
    def __init__(self,  num_points, radius=5, center=[0,0], drawing=drawing_global, angle=0):
        self._num_points = num_points
        self._radius = radius
        self._center = center
        self.drawing = drawing
        self._angle = angle
        self.points = self.polygon()
        self.fractal_points = list(self.points)
        self.rotate = None
    
    def polygon(self):
        polygon_angle = 360 / self._num_points # In degrees
        points = []
        for i in range(self._num_points):
            current_angle = polygon_angle * i - 90 + self._angle
            x = self._center[0] + self._radius * math.cos(current_angle*(math.pi/180)) # Convert to rad for cos and sin
            y = self._center[1] + self._radius * math.sin(current_angle*(math.pi/180))
            points.append([x, y])
        return points
        
    def draw(self):
        self.drawing.add(self.drawing.polygon(self.points))

    def rotate(self, rotation_angle:float):
        self._angle += rotation_angle
        self.points = self.polygon()
                
    def draw_outline(self, outline_offset):
        outline_polygon = Polygon(self.num_points, self.radius + outline_offset, self.center, self.drawing, self.angle)
        self.drawing.add(self.drawing.polygon(outline_polygon.points))
        
    def draw_fractal(self, shrinkage:float, depth:int,  rotate=False, radius=None, first=True,):
        # print(f"self.fractal_points is {self.fractal_points}")
        if first: 
            radius = self.radius*shrinkage
            self.draw()
            self.rotate = rotate
        else: radius = radius*shrinkage
        if depth >= 1:
            current_fractal_points = []
            for point in self.fractal_points:
                fractal_polygon = Polygon(self.num_points, radius, point, self.drawing, rotate)
                fractal_polygon.draw()
                for fractal_point in fractal_polygon.points:
                    current_fractal_points.append(fractal_point)
            self.fractal_points = list(current_fractal_points)
            self.draw_fractal(shrinkage, depth-1, rotate+self.rotate, radius, False)
        else:
            # print(self.fractal_points)
            return self.fractal_points
        
    @property
    def num_points(self):
        return self._num_points
    @num_points.setter
    def num_points(self, num_points:int):
        self._num_points = num_points
        self.points = self.polygon()
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, radius:float):
        self._radius = radius
        self.points = self.polygon()
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self, center: List[float]):
        if len(center) == 2:
            self._center = center
            self.points = self.polygon()
        else:
            print(f"Center must be list of length 2, defining x and y coordinates") 
    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, angle:float):
        self._angle = angle
        self.points = self.polygon()

class Grid:
    def __init__(self, spacing, polygon, num_x, num_y, origin=[0, 0], drawing=drawing_global):
        self._spacing = spacing
        self._num_x = num_x
        self._num_y = num_y
        self._polygon = polygon
        self._origin = origin
        self.drawing = drawing
        self.points = self._grid()
        self.polygons = self._generate_polygons()
        self.polygon_points = self._polygon_points()
        
    def _grid(self):
        grid = []
        for i in range(self._num_y):
            row = []
            for j in range(self._num_x):
                x = j * self._spacing
                y = self.origin[1] + i * self._spacing
                row.append([x, y])
            grid.append(row)
        return grid
    
    def _generate_polygons(self):
        polygons = []
        for i, row in enumerate(self.points):
            polygon_row = []
            for j, coord in enumerate(row):
                polygon = Polygon(self._polygon.num_points, self._polygon.radius, coord, self.drawing, self._polygon.angle)
                polygon_row.append(polygon)
            polygons.append(polygon_row)
        self.polygons = polygons
        self.polygon_points = self._polygon_points()
        return polygons
    
    def _polygon_points(self):
        polygon_points = [] # Flat list of all points of all polygons in grid
        for i, row in enumerate(self.polygons):
            for j, polygon in enumerate(row):
                for point in polygon.points:
                    polygon_points.append(point)
        return polygon_points
    
    def modify_polygons(self, callback, **kwargs):
        if self.num_x == 1 and self.num_y == 1: 
            raise ValueError("Cannot modify grid of size 1 x 1")
        else:
            for i, row in enumerate(self.polygons):
                for j, polygon in enumerate(row):
                    callback(self, polygon, i, j, **kwargs)
                  
    def draw_polygons(self):
        self.modify_polygons(lambda self, polygon, i, j: self.drawing.add(self.drawing.polygon(polygon.points)))
        
    def center_polygon(self):
        center_i_j = self.center_i_j()
        return self.polygons[center_i_j[0]][center_i_j[1]]
    
    def center_geometric(self):
        center_x = (self.points[0][0][0] + self.points[0][0][-1]) / 2 # Left column x to right column x
        center_y = (self.points[0][0][1] + self.points[-1][0][1]) / 2 # Top row y to bottom row y
        # center = Polygon(3,1, [center_x, center_y]) # Make tiny triangle to show center
        # self.drawing.add(self.drawing.polygon(center.points))
        return [center_x, center_y]
    
    def center_i_j(self):
        center_polygon_j = math.ceil(self._num_x/2)-1
        center_polygon_i = math.ceil(self._num_y/2)-1
        return [center_polygon_i, center_polygon_j]
    
    def distance(self, point_1:list, point_2:list):
        return math.sqrt( (point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 )
    
    def max_distance(self, origin_point):
        distances = []
        top_points = self.points[0]
        bottom_points = self.points[-1]
        left_points = [ row[0] for row in self.points ]
        right_points = [row[-1] for row in self.points]
        edge_points = top_points + bottom_points + left_points + right_points
        for point in edge_points:
            distances.append(self.distance(origin_point, point))
        return max(distances)
            
    def draw_outlines(self, outline_offset):
        self.modify_polygons(lambda self, polygon, i, j: polygon.draw_outline(outline_offset))
    
    @property
    def spacing(self):
        return self._spacing
    
    @spacing.setter
    def spacing(self, spacing:float):
        self._spacing = spacing
        self.points = self._grid()
        
    @property
    def num_x(self):
        return self._num_x
    
    @num_x.setter
    def num_x(self, num_x:int):
        self._num_x = num_x
        self.points = self._grid()
        
    @property
    def num_y(self):
        return self._num_y
    
    @num_y.setter
    def num_y(self, num_y:int):
        self._num_y = num_y
        self.points = self._grid()
        
    @property
    def polygon(self):
        return self._polygon
    
    @polygon.setter
    def polygon(self, polygon):
        self._polygon = polygon
        self._generate_polygons()
        
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, origin:List):
        self._origin = origin
        self.points = self._grid()
        
class GridIsometric(Grid):
    def __init__(self, spacing, num_x, num_y, polygon, symmetric=True, origin=[0, 0], drawing=drawing_global):
        self._spacing = spacing
        self._num_x = num_x
        self._num_y = num_y
        self._polygon = polygon
        self._origin = origin
        self.drawing = drawing
        self.points = self._grid()
        if symmetric: self._grid_symmetric()
        self.polygons = self._generate_polygons()
        self.polygon_points = self._polygon_points()
        
    def _grid(self):
        spacing_y = self._spacing * math.sin(60*(math.pi/180))
        grid = []
        for i in range(self._num_y):
            row = []
            for j in range(self._num_x):
                if i % 2 == 0:
                    x = self.origin[0] + j * self._spacing
                else:
                    x = j * self._spacing + (self._spacing/2)
                y = self.origin[1] + i * spacing_y
                row.append([x, y])
            grid.append(row)
        return grid
    
    def center_geometric(self):
        center_x = (self.points[0][0][0] + self.points[1][-1][0]) / 2 # Center of length from first row first point to second row last point (it's isometric so second row is shifted over)
        center_y = (self.points[0][0][1] + self.points[-1][0][1]) / 2 # Top row y to bottom row y, colum doesn't matter
        # center = Polygon(3,1, [center_x, center_y]) # Make tiny triangle to show center
        # self.drawing.add(self.drawing.polygon(center.points))
        return [center_x, center_y]
    
    def _grid_symmetric(self):
        ### Adds an extra point to the non-shifted rows to make grid bilaterally symmetric. Better for making fractal grids
        for i, row in enumerate(self.points):
            if i%2 == 0:
                last_point = row[-1]
                next_x = last_point[0] + self.spacing
                next_y = last_point[1]
                self.points[i].append([next_x, next_y])
        self.polygons = self._generate_polygons()
        return 

class GridMandala(Grid):
    def __init__(self, radius, symmetry, num_y, polygon, origin=[0, 0], drawing=drawing_global):
        self.radius = radius
        
        

class Mandala: # TODO consider making this inherit from Grid class
    def __init__(self, drawing, mandala_radius:float, symmetry:int, polygon:Polygon, angle=0, center=[0, 0]):
        self.drawing = drawing
        self._radius = mandala_radius
        self._symmetry = symmetry
        self._polygon = polygon
        self._angle = angle
        self._center = center
        self.points = self._mandala()
        self._polygons = self._generate_polygons()
    
    def _mandala(self):
        mandala_angle = 360 / self._symmetry # In degrees
        points = []
        for i in range(self._symmetry):
            current_angle = mandala_angle * i - 90 + self._angle
            x = self._center[0] + self._radius * math.cos(current_angle*(math.pi/180)) # Convert to rad for cos and sin
            y = self._center[1] + self._radius * math.sin(current_angle*(math.pi/180))
            points.append([x, y])
        return points
    
    def _generate_polygons(self):
        polygons = []
        mandala_angle = 360 / self._symmetry # In degrees
        for i, coord in enumerate(self.points):
            current_angle = mandala_angle * i + self._angle
            # print(f"in mandala _generate_polygons and i is {i}, mandala_angle is {mandala_angle} and current_angle is  {current_angle}")
            polygon = Polygon(self._polygon.num_points, self._polygon.radius, coord, self.drawing, current_angle)
            polygons.append(polygon)
        self._polygons = polygons
        return polygons
    
    def modify_polygons(self, callback, **kwargs):
        for i, polygon in enumerate(self._polygons):
            callback(self, polygon, i, **kwargs)
                  
    def draw_polygons(self):
        self.modify_polygons(lambda self, polygon, i: self.drawing.add(self.drawing.polygon(polygon.points))) 
        
    def draw_outlines(self, outline_offset):
        self.modify_polygons(lambda self, polygon, i: polygon.draw_outline(outline_offset))
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, radius:float):
        self._radius = radius
        self.points = self.points()
        
    @property
    def symmetry(self):
        return self._symmetry
    
    @symmetry.setter
    def radius(self, symmetry:float):
        self._symmetry = symmetry
        self.points = self.points()
        
    @property
    def polygon(self):
        return self._polygon
    
    @polygon.setter
    def polygon(self, polygon):
        self._polygon = polygon
        self._generate_polygons()
        
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, angle:float):
        self._angle = angle
        self.points = self.polygon()
        
    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, center:List):
        self._center = center
        self.points = self.points()
        
def _polygon_points(item):
    """Return list of [x,y] for a layer item (Polygon or list of points)."""
    if hasattr(item, "points"):
        return list(item.points)
    return list(item)


def _canonical_edge(p1, p2, ndigits=6):
    """Normalize edge so (a,b) and (b,a) compare equal for counting."""
    a = (round(float(p1[0]), ndigits), round(float(p1[1]), ndigits))
    b = (round(float(p2[0]), ndigits), round(float(p2[1]), ndigits))
    return (min(a, b), max(a, b))


def _centroid_of_vertices(vertices):
    """Return [cx, cy] for the centroid of a vertex list."""
    if not vertices:
        return None
    n = len(vertices)
    cx = sum(p[0] for p in vertices) / n
    cy = sum(p[1] for p in vertices) / n
    return [cx, cy]


def _polygon_from_vertices(vertices, num_points, drawing):
    """Build a Polygon at the centroid of the vertex list with radius = mean distance to centroid (for modify_polygons)."""
    if len(vertices) < 3 or drawing is None:
        return None
    cx = sum(p[0] for p in vertices) / len(vertices)
    cy = sum(p[1] for p in vertices) / len(vertices)
    radii = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in vertices]
    radius = sum(radii) / len(radii) if radii else 0
    return Polygon(num_points, radius, [cx, cy], drawing, 0)


class EdgeMandala:
    """
    Builds layers of polygons: each new layer has one polygon per *outer* edge of
    the previous layer. The new polygon shares that edge (same two vertices) and
    is placed on the outside (away from the mandala center).
    Layers store the actual generated shapes (vertex lists; layer 0 is the seed Polygon).
    A separate grid (2D list of [x,y] points) holds the center of every polygon so you
    can draw other polygons at those points (e.g. draw_polygons_at_grid) and optionally
    make them face the mandala center.
    """
    def __init__(self, seed_polygon):
        self._seed_polygon = seed_polygon
        self.drawing = getattr(seed_polygon, "drawing", None)
        self.layers = [[self.seed_polygon]]
        # Grid of center points: grid[layer_i][poly_j] = [x, y] center of that polygon.
        self.grid = [[list(self.seed_polygon.center)]]
        # Polygon object per cell (for modify_polygons); same shape as grid/layers.
        self.polygons = [[self.seed_polygon]]

    def generate_edge_polygon(self, edge, num_points, interior_point=None):
        start, end = edge[0], edge[1]
        return create_edge_polygon(start, end, num_points, interior_point=interior_point)

    def _outer_edges_and_centroid(self, layer):
        """Collect all edges from the layer, count occurrences; return outer edges and centroid."""
        edge_counts = Counter()
        edge_to_endpoints = {}  # canonical -> (start, end) as lists
        all_x, all_y, total = 0.0, 0.0, 0
        for item in layer:
            pts = _polygon_points(item)
            n = len(pts)
            for idx in range(n):
                p1 = pts[idx]
                p2 = pts[(idx + 1) % n]
                key = _canonical_edge(p1, p2)
                edge_counts[key] += 1
                edge_to_endpoints[key] = ([float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])])
            for p in pts:
                all_x += p[0]
                all_y += p[1]
                total += 1
        centroid = [all_x / total, all_y / total] if total else None
        outer = [edge_to_endpoints[k] for k in edge_counts if edge_counts[k] == 1]
        return outer, centroid

    def generate_layer(self, num_points=None):
        """
        Add one new layer: a polygon on each outer edge of the previous layer (shared edge, outward).
        num_points: number of vertices for each new polygon in this layer (e.g. 3 = triangles).
        If None, uses the seed polygon's num_points.
        """
        last_layer = self.layers[-1]
        outer_edges, interior_point = self._outer_edges_and_centroid(last_layer)
        if not outer_edges:
            return
        n_pts = num_points if num_points is not None else self.seed_polygon.num_points
        center = self.center()
        new_layer = []
        new_grid_row = []
        new_polygons_row = []
        for (start, end) in outer_edges:
            vertices = self.generate_edge_polygon((start, end), n_pts, interior_point=interior_point)
            new_layer.append(vertices)
            pt = _centroid_of_vertices(vertices)
            if pt is not None:
                new_grid_row.append(pt)
            poly = _polygon_from_vertices(vertices, n_pts, self.drawing)
            if poly is not None:
                # Default: one edge faces the mandala center (angle so edge midpoint points at center).
                poly.angle = (
                    math.degrees(math.atan2(center[1] - poly.center[1], center[0] - poly.center[0]))
                    + 90
                    - 180 / n_pts
                )
                new_polygons_row.append(poly)
        self.layers.append(new_layer)
        self.grid.append(new_grid_row)
        self.polygons.append(new_polygons_row)

    def generate_outer_layer(self, num_points=None):
        """
        Add one new layer using only the *outermost* outer edges of the previous layer:
        edges whose distance from the layer centroid is maximum. If several edges tie
        for that maximum distance, all of them get a new polygon (same as generate_layer
        but filtered to farthest edges only).
        num_points: number of vertices for each new polygon; if None, uses seed num_points.
        """
        last_layer = self.layers[-1]
        outer_edges, interior_point = self._outer_edges_and_centroid(last_layer)
        if not outer_edges or interior_point is None:
            return
        cx, cy = interior_point[0], interior_point[1]
        max_dist_sq = -1.0
        for (start, end) in outer_edges:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            d_sq = (cx - mx) ** 2 + (cy - my) ** 2
            if d_sq > max_dist_sq:
                max_dist_sq = d_sq
        if max_dist_sq < 0:
            return
        tol = 1e-9 * max(max_dist_sq, 1.0)
        outermost = []
        for (start, end) in outer_edges:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            d_sq = (cx - mx) ** 2 + (cy - my) ** 2
            if abs(d_sq - max_dist_sq) <= tol:
                outermost.append((start, end))
        if not outermost:
            return
        n_pts = num_points if num_points is not None else self.seed_polygon.num_points
        center = self.center()
        new_layer = []
        new_grid_row = []
        new_polygons_row = []
        for (start, end) in outermost:
            vertices = self.generate_edge_polygon((start, end), n_pts, interior_point=interior_point)
            new_layer.append(vertices)
            pt = _centroid_of_vertices(vertices)
            if pt is not None:
                new_grid_row.append(pt)
            poly = _polygon_from_vertices(vertices, n_pts, self.drawing)
            if poly is not None:
                # Default: one edge faces the mandala center.
                poly.angle = (
                    math.degrees(math.atan2(center[1] - poly.center[1], center[0] - poly.center[0]))
                    + 90
                    - 180 / n_pts
                )
                new_polygons_row.append(poly)
        self.layers.append(new_layer)
        self.grid.append(new_grid_row)
        self.polygons.append(new_polygons_row)

    def generate_layers(self, n_or_num_points):
        """
        Add new layers (polygons on outer edges of the previous layer).
        n_or_num_points: int or list of ints.
          - int n: add n layers, each using the seed polygon's num_points.
          - list of ints: add len(list) layers; layer i uses list[i] as num_points (e.g. [3, 4, 5] = triangles, then quads, then pentagons).
        """
        if isinstance(n_or_num_points, list):
            for num_points in n_or_num_points:
                self.generate_layer(num_points=num_points)
        else:
            for _ in range(n_or_num_points):
                self.generate_layer()

    def center(self):
        """Return the mandala center [x, y] (seed polygon center)."""
        return list(self.seed_polygon.center)

    def center_polygon(self):
        """Return the polygon at the center (seed, layer 0). For use with morphs like ripple_morph."""
        return self.polygons[0][0]

    def max_distance(self, origin_point):
        """Max distance from origin_point to any polygon center (for morphs like ripple_morph)."""
        best = 0.0
        for row in self.polygons:
            for polygon in row:
                d = math.sqrt(
                    (origin_point[0] - polygon.center[0]) ** 2
                    + (origin_point[1] - polygon.center[1]) ** 2
                )
                if d > best:
                    best = d
        return best

    def modify_polygons(self, callback, **kwargs):
        """Call callback(self, polygon, layer_index, polygon_index, **kwargs) for each polygon (like Mandala/Grid)."""
        for i, row in enumerate(self.polygons):
            for j, polygon in enumerate(row):
                callback(self, polygon, i, j, **kwargs)

    def draw_polygons(self):
        """Draw each polygon in self.polygons (so modifications from modify_polygons are visible)."""
        if self.drawing is None:
            return
        for row in self.polygons:
            for polygon in row:
                self.drawing.add(self.drawing.polygon(polygon.points))

    def draw_polygons_at_grid(self, num_points, radius, drawing=None, face_center=False):
        """
        Draw a polygon at every point in self.grid (center of every layer polygon).
        num_points, radius: shape of each polygon.
        If face_center is True, rotate each polygon so it faces the mandala center.
        """
        drawing = drawing or self.drawing
        if drawing is None:
            return
        center = self.center()
        cx, cy = center[0], center[1]
        for row in self.grid:
            for pt in row:
                px, py = pt[0], pt[1]
                if face_center:
                    angle_rad = math.atan2(cy - py, cx - px)
                    angle_deg = math.degrees(angle_rad) + 90
                else:
                    angle_deg = 0
                poly = Polygon(num_points, radius, [px, py], drawing, angle_deg)
                drawing.add(drawing.polygon(poly.points))

    def draw_layers(self, drawing=None):
        """Draw all layers to the given drawing (default: seed_polygon's drawing)."""
        drawing = drawing or self.drawing
        if drawing is None:
            return
        for layer in self.layers:
            for item in layer:
                pts = _polygon_points(item)
                if len(pts) >= 3:
                    drawing.add(drawing.polygon(pts))

    @property
    def seed_polygon(self):
        return self._seed_polygon

    @seed_polygon.setter
    def seed_polygon(self, polygon):
        self._seed_polygon = polygon
        
    
def create_edge_polygon(start, end, num_vertices, interior_point=None):
    """
    Create a regular polygon that has (start, end) as one shared edge.
    The polygon is placed on the side of the edge opposite to interior_point
    (so for a mandala, interior_point is the center and the new polygon goes outward).
    Returns list of [x, y] vertices: [start, end, ...] so the edge is shared.
    """
    start = [float(start[0]), float(start[1])]
    end = [float(end[0]), float(end[1])]
    if num_vertices < 3:
        return [start, end]
    ex = end[0] - start[0]
    ey = end[1] - start[1]
    line_length = math.sqrt(ex * ex + ey * ey)
    if line_length < 1e-10:
        return [start, end]
    # Perpendicular unit vector (one of the two sides).
    px = -ey / line_length
    py = ex / line_length
    midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
    # Choose side: polygon center on the side opposite to interior_point.
    if interior_point is not None:
        dx = interior_point[0] - midpoint[0]
        dy = interior_point[1] - midpoint[1]
        if dx * px + dy * py > 0:
            px, py = -px, -py
    # For a regular n-gon with one edge of length L: circumradius R = L / (2*sin(pi/n)),
    # distance from edge midpoint to center = R*cos(pi/n) = L/(2*tan(pi/n)).
    n = num_vertices
    half_tan = math.tan(math.pi / n)
    dist_to_center = line_length / (2 * half_tan) if half_tan > 1e-10 else 0
    center = [midpoint[0] + px * dist_to_center, midpoint[1] + py * dist_to_center]
    # Circumradius so that start and end lie on the circle.
    R = line_length / (2 * math.sin(math.pi / n)) if n > 0 else 0
    # Angle from center to start.
    angle0 = math.atan2(start[1] - center[1], start[0] - center[0])
    angle_step = 2 * math.pi / n
    vertices = []
    for k in range(n):
        angle = angle0 + k * angle_step
        x = center[0] + R * math.cos(angle)
        y = center[1] + R * math.sin(angle)
        vertices.append([x, y])
    return vertices
    
def radius_morph_polygon_center(grid, polygon:Polygon, i:int, j:int, magnitude):
    center = grid.center_polygon().center
    difference_x = abs(center[0] - polygon.center[0])
    difference_y = abs(center[1] - polygon.center[1])
    
    polygon.radius -= (difference_x + difference_y)*magnitude/10

def circle_morph(grid:GridIsometric, polygon:Polygon, i:int, j:int, magnitude:float, decrease_out:bool = True):
    center = grid.center_polygon().center
    # Try every distance from center to each corner and get max
    max_distance = grid.max_distance(center)
    normalize = polygon.radius/max_distance
    difference = math.sqrt( (center[0] - polygon.center[0])**2 + (center[1] - polygon.center[1])**2 )
    if decrease_out: polygon.radius -= difference*normalize*magnitude
    else: polygon.radius = 0 + difference*normalize*magnitude
        
def ripple_morph(
    grid: GridIsometric,
    polygon: Polygon,
    i: int,
    j: int,
    magnitude: float,
    decrease_out: bool = True,
    num_waves: float = 5.0,
    decay_rate: float = 1.5,
):
    """
    Morph polygon radius by a sine (water-ripple) effect: oscillation with fixed spatial
    frequency, and decay so amplitude shrinks with distance from grid center.
    Same signature pattern as circle_morph; optional num_waves (ripple cycles from center to edge)
    and decay_rate (how fast amplitude falls off with distance).
    """
    center = grid.center_polygon().center
    max_distance = grid.max_distance(center)
    distance = math.sqrt(
        (center[0] - polygon.center[0]) ** 2 + (center[1] - polygon.center[1]) ** 2
    )
    # Phase: num_waves full cycles from center to max_distance (frequency constant in space).
    phase = 2 * math.pi * num_waves * distance / max_distance if max_distance > 0 else 0
    # Decay: amplitude falls off with distance (1 at center, smaller toward edge).
    decay_factor = math.exp(-decay_rate * distance / max_distance) if max_distance > 0 else 1.0
    # Amplitude on the order of radius*magnitude (like circle_morph) so the ripple is visible.
    ripple = magnitude * polygon.radius * math.sin(phase) * decay_factor * 0.5
    if decrease_out:
        polygon.radius -= ripple
    else:
        polygon.radius += ripple
        
def radius_morph_2(grid, polygon:Polygon, i:int, j:int):
    center_x = (grid.num_x - 1)/2
    center_y = (grid.num_y - 1)/2
    difference_x = abs(center_x - j)
    difference_y = abs(center_y - i)
    # print(f"i is {i}, j is {j}, center_x is {center_x}, difference_x is {difference_x}")
    polygon.radius -= (difference_x + difference_y)/4

def sine_morph(grid, polygon:Polygon, i:int, j:int, num_waves=1, amplitude=None):
    if amplitude == None: amplitude = grid.num_x/2
    frequency = num_waves / grid.num_y
    sine_value = amplitude * math.sin(2 * math.pi * frequency * i) + (grid.num_x/2)
    difference = abs(sine_value - j)
    polygon.radius += (difference/10)
    polygon.angle += difference*3 
    
def linear_gradient(grid:GridIsometric, polygon:Polygon, i:int, j:int,magnitude:float, angle, decrease_out:bool=False):
    radians = -angle*(math.pi/180) # Angle should go counterclockwise
    center = grid.center_polygon().center
    distance = abs(math.cos(radians)*(center[1]-polygon.center[1]) - math.sin(radians)*(center[0]-polygon.center[0]))
    max_distance = grid.max_distance(center)
    normalize = polygon.radius / max_distance
    if decrease_out: polygon.radius -= distance*normalize*magnitude
    else:polygon.radius = 0 + distance*normalize*magnitude


# Minimum area for a region to be kept (avoids degenerate slivers from floating point).
_MIN_REGION_AREA = 1e-10


def _polygon_points_from_drawing(drawing: svgwrite.Drawing) -> List[List[List[float]]]:
    """
    Extract polygon vertex lists from an svgwrite Drawing.
    Returns a list of polygons; each polygon = list of [x, y] vertex coordinates.
    Approach: get the drawing as XML, iterate over every element, find <polygon> nodes,
    parse the "points" attribute into numbers, and build [x,y] pairs for each polygon.
    """
    start_time = time.perf_counter()
    # List of polygons; each polygon is a list of [x, y] points (vertices).
    all_polygon_vertices = []
    drawing_xml_root = drawing.get_xml()
    for element in drawing_xml_root.iter():
        # Handle SVG namespace: tag may be "polygon" or "{uri}polygon".
        tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag_name != "polygon":
            continue
        # SVG polygon has a "points" attribute: "x1,y1 x2,y2 x3,y3 ..."
        points_attr = element.get("points")
        if not points_attr:
            continue
        # Parse into floats: split on space and comma, then take pairs as x,y.
        flat_numbers = []
        for token in points_attr.strip().replace(",", " ").split():
            try:
                flat_numbers.append(float(token))
            except ValueError:
                continue
        # Need at least 3 vertices (6 numbers) and an even count for x,y pairs.
        if len(flat_numbers) >= 6 and len(flat_numbers) % 2 == 0:
            vertex_list = [
                [flat_numbers[index], flat_numbers[index + 1]]
                for index in range(0, len(flat_numbers), 2)
            ]
            all_polygon_vertices.append(vertex_list)
    elapsed = time.perf_counter() - start_time
    print(f"[timer] _polygon_points_from_drawing: {elapsed:.4f} s")
    return all_polygon_vertices


def _ensure_valid_polygons(geometries):
    """
    Return a flat list of valid Shapely Polygon(s) for set ops (e.g. symmetric_difference_all).
    Invalid or non-polygon geoms are fixed with buffer(0) and exploded into polygons.
    """
    out = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        for poly in _to_polygon_list(geom):
            if poly.is_empty or poly.area <= 0:
                continue
            out.append(poly)
    return out


def _to_polygon_list(geometry):
    """
    Flatten a Shapely geometry into a list of Polygon(s).
    A region = one Shapely Polygon (no overlaps). MultiPolygon/Collection become multiple items.
    Approach: check geometry type; if Polygon return it as a single-element list;
    if MultiPolygon, collect each sub-polygon; if GeometryCollection, recurse into each part.
    """
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [single_polygon for single_polygon in geometry.geoms if not single_polygon.is_empty]
    if geometry.geom_type == "GeometryCollection":
        polygons_out = []
        for sub_geometry in geometry.geoms:
            polygons_out.extend(_to_polygon_list(sub_geometry))
        return polygons_out
    return []


def _shapely_polygons_to_points_lists(shapely_polygons):
    """
    Convert a list of Shapely Polygons into points_lists (list of vertex lists).
    Each item = one polygon as [[x,y], [x,y], ...]. Used when we have Shapely
    polygons in memory (e.g. from a grid) and need to pass them into arrangement builders.
    Approach: loop through each Shapely polygon, read its exterior ring coordinates,
    and build a vertex list [x,y] for each; skip empty or degenerate polygons.
    """
    start_time = time.perf_counter()
    points_lists = []
    for polygon in shapely_polygons:
        if polygon is None or polygon.is_empty:
            continue
        # Exterior ring: closed sequence of (x, y) coordinates.
        exterior_coords = list(polygon.exterior.coords)
        if len(exterior_coords) < 3:
            continue
        vertex_list = [[float(point[0]), float(point[1])] for point in exterior_coords]
        points_lists.append(vertex_list)
    elapsed = time.perf_counter() - start_time
    print(f"[timer] _shapely_polygons_to_points_lists: {elapsed:.4f} s")
    return points_lists


def _points_lists_to_shapely_polys(points_lists):
    """
    Convert points_lists (list of vertex lists) into a flat list of Shapely Polygons.
    Each polygon is closed and validated (invalid ones are fixed with buffer(0)).
    A region = one Shapely Polygon (no overlaps). After buffer(0) one input can become
    several valid polygons; we flatten them into the result list.
    Approach: loop through each vertex list, close the ring if needed, build a Shapely
    polygon; fix invalid ones with buffer(0), then collect every resulting polygon piece.
    """
    shapely_polygons = []
    for vertex_list in points_lists:
        if len(vertex_list) < 3:
            continue
        # Close the ring if the first and last point differ.
        if vertex_list[0] != vertex_list[-1]:
            vertex_list = list(vertex_list) + [vertex_list[0]]
        polygon = ShapelyPolygon(vertex_list)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        # One Shapely poly might become several (e.g. after buffer(0)); flatten them.
        for polygon_piece in _to_polygon_list(polygon):
            if not polygon_piece.is_empty and polygon_piece.area > 0:
                shapely_polygons.append(polygon_piece)
    return shapely_polygons


def _build_arrangement(points_lists):
    """
    Build the planar arrangement: decompose overlapping polygons into non-overlapping
    regions (no depth tracking). Returns list of Shapely Polygons or None if no input.
    A region = one contiguous area that belongs to exactly one “layer” of the arrangement
    (no overlaps); we don’t track how many shapes cover it. Used by fracture().
    Approach: start with the first polygon as initial regions; for each later polygon we
    split every existing region into “outside this polygon” and “inside this polygon”,
    then add the part of the new polygon not yet covered by any region. Repeat until
    all input polygons are processed.
    """
    start_time = time.perf_counter()
    shapely_polys = _points_lists_to_shapely_polys(points_lists)
    if not shapely_polys:
        return None

    # Regions = list of non-overlapping Shapely Polygons. Start with the first input polygon.
    regions = [
        region for region in _to_polygon_list(shapely_polys[0])
        if region.area > _MIN_REGION_AREA
    ]

    for current_polygon in shapely_polys[1:]:
        new_regions = []
        for region in regions:
            # Region outside current polygon: keep as-is (difference).
            new_regions.extend(
                piece for piece in _to_polygon_list(region.difference(current_polygon))
                if not piece.is_empty and piece.area > _MIN_REGION_AREA
            )
            # Region inside current polygon: intersection.
            new_regions.extend(
                piece for piece in _to_polygon_list(region.intersection(current_polygon))
                if not piece.is_empty and piece.area > _MIN_REGION_AREA
            )
        # Part of current polygon not yet covered by any region: add as new regions.
        union_existing = unary_union(regions)
        uncovered_part = current_polygon.difference(union_existing)
        new_regions.extend(
            piece for piece in _to_polygon_list(uncovered_part)
            if not piece.is_empty and piece.area > _MIN_REGION_AREA
        )
        regions = new_regions

    elapsed = time.perf_counter() - start_time
    print(f"[timer] _build_arrangement: {elapsed:.4f} s")
    return regions


def _build_arrangement_with_depths(points_lists):
    """
    Build the planar arrangement of all input polygons and track depth per region.
    A region = one contiguous non-overlapping polygon. Depth = how many input shapes
    cover that region (1 = one shape, 2 = two overlap, etc.). Overlaps = depth - 1.

    Algorithm:
    - Start with the first polygon as a single region at depth 1 (one shape covers it).
    - For each additional polygon P: split every existing region R into
      (R outside P) and (R inside P). Outside keeps same depth; inside gets depth+1.
      Then add the part of P that is not yet covered by any region, at depth 1.
    - No point-in-polygon tests: depth is computed during the split, O(1) per region.

    Returns (regions, depths) or (None, None) if no valid input.
    """
    start_time = time.perf_counter()
    shapely_polys = _points_lists_to_shapely_polys(points_lists)
    if not shapely_polys:
        return None, None

    # First polygon: all of it is one region at depth 1.
    regions = _to_polygon_list(shapely_polys[0])
    depths = [1] * len(regions)

    # Filter by minimum area (keep regions and depths in sync).
    regions, depths = zip(*[
        (region, depth) for region, depth in zip(regions, depths)
        if region.area > _MIN_REGION_AREA
    ])
    regions = list(regions)
    depths = list(depths)

    # Add each subsequent polygon: split existing regions and assign depths.
    for polygon_index in range(1, len(shapely_polys)):
        current_polygon = shapely_polys[polygon_index]
        new_regions = []
        new_depths = []

        for existing_region, region_depth in zip(regions, depths):
            # Part of existing_region outside current_polygon: same depth (not covered by P).
            difference_parts = existing_region.difference(current_polygon)
            for piece in _to_polygon_list(difference_parts):
                if not piece.is_empty and piece.area > _MIN_REGION_AREA:
                    new_regions.append(piece)
                    new_depths.append(region_depth)
            # Part of existing_region inside current_polygon: depth increases by one.
            intersection_parts = existing_region.intersection(current_polygon)
            for piece in _to_polygon_list(intersection_parts):
                if not piece.is_empty and piece.area > _MIN_REGION_AREA:
                    new_regions.append(piece)
                    new_depths.append(region_depth + 1)

        # Part of current polygon not yet covered by any region: depth 1 (only this shape).
        union_existing = unary_union(regions)
        uncovered_part = current_polygon.difference(union_existing)
        for piece in _to_polygon_list(uncovered_part):
            if not piece.is_empty and piece.area > _MIN_REGION_AREA:
                new_regions.append(piece)
                new_depths.append(1)

        regions = new_regions
        depths = new_depths

    elapsed = time.perf_counter() - start_time
    print(f"[timer] _build_arrangement_with_depths: {elapsed:.4f} s")
    return regions, depths


def _draw_regions(drawing: svgwrite.Drawing, regions) -> None:
    """
    Clear the drawing and add one polygon path per region (exterior ring only).
    A region = one Shapely Polygon; we draw its boundary as an SVG polygon.
    Approach: clear existing elements, then loop through each region, get its
    exterior coordinates, close the ring if needed, and add a polygon to the drawing.
    """
    start_time = time.perf_counter()
    drawing.elements.clear()
    for region_polygon in regions:
        ring = region_polygon.exterior
        if ring is None:
            continue
        coords = list(ring.coords)
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        points = [[float(coord[0]), float(coord[1])] for coord in coords]
        drawing.add(drawing.polygon(points))
    elapsed = time.perf_counter() - start_time
    print(f"[timer] _draw_regions: {elapsed:.4f} s")


def fracture(
    drawing: svgwrite.Drawing, polygons=None
) -> svgwrite.Drawing:
    """
    Fracture: decompose overlapping polygons into non-overlapping regions
    and draw each region as a separate path (adjacent edges, no overlaps).
    Same as Inkscape's Path > Fracture.
    A region = one contiguous non-overlapping polygon in the arrangement.
    Approach: get polygon vertex lists (from argument or from the drawing), build the
    planar arrangement with _build_arrangement(), then draw each region as one SVG path.
    Either pass a list of Shapely Polygons (e.g. grid.get_shapely_polygons())
    or leave polygons=None to extract polygons from the drawing. Modifies the drawing in place.
    """
    start_time = time.perf_counter()
    if polygons is not None:
        points_lists = _shapely_polygons_to_points_lists(polygons)
    else:
        points_lists = _polygon_points_from_drawing(drawing)
    if not points_lists:
        return drawing

    regions = _build_arrangement(points_lists)
    if not regions:
        return drawing

    _draw_regions(drawing, regions)
    elapsed = time.perf_counter() - start_time
    print(f"[timer] fracture (total): {elapsed:.4f} s")
    return drawing


def _safe_symmetric_difference(a, b):
    """
    a.symmetric_difference(b) with GEOSException handling: retry after buffer(0) on
    invalid geometry so iterative XOR can complete on messy input.
    """
    try:
        return a.symmetric_difference(b)
    except GEOSException:
        pass
    a = a.buffer(0) if not a.is_valid or a.is_empty else a
    b = b.buffer(0) if not b.is_valid or b.is_empty else b
    if a.is_empty:
        return b
    if b.is_empty:
        return a
    try:
        return a.symmetric_difference(b)
    except GEOSException:
        pass
    return a.buffer(0).symmetric_difference(b.buffer(0))


def _iterative_symmetric_difference_chunk(polygon_list):
    """
    Worker for parallel symmetric_difference: compute iterative XOR for one chunk of
    Shapely polygons. Used from ThreadPoolExecutor (threads, not processes) so no
    pickling or if __name__ == '__main__' guard is needed in the caller.
    Returns a single Shapely geometry or None if polygon_list is empty.
    """
    if not polygon_list:
        return None
    result = polygon_list[0]
    for current_polygon in polygon_list[1:]:
        result = _safe_symmetric_difference(result, current_polygon)
        if result is not None and not result.is_empty and not result.is_valid:
            result = result.buffer(0)
    return result


def _xor_shapely_polygons(shapely_polys, parallel: bool = False):
    """
    XOR of a list of Shapely polygons. Tries symmetric_difference_all first;
    on GEOSException falls back to iterative XOR (sequential or parallel).
    Returns the result geometry or None if empty/failed.
    """
    if not shapely_polys:
        return None
    result = None
    if symmetric_difference_all is not None:
        try:
            result = symmetric_difference_all(shapely_polys)
        except GEOSException:
            pass
    if result is None:
        if parallel:
            max_workers = min(os.cpu_count() or 4, len(shapely_polys))
            if max_workers <= 1:
                result = _iterative_symmetric_difference_chunk(shapely_polys)
            else:
                chunks = [
                    [shapely_polys[i] for i in range(w, len(shapely_polys), max_workers)]
                    for w in range(max_workers)
                ]
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    partial_results = list(executor.map(_iterative_symmetric_difference_chunk, chunks))
                result = None
                for partial in partial_results:
                    if partial is not None:
                        result = partial if result is None else _safe_symmetric_difference(result, partial)
        else:
            result = _iterative_symmetric_difference_chunk(shapely_polys)
    return result


def symmetric_difference(
    drawing: svgwrite.Drawing, polygons=None, keep_even: bool = False
) -> svgwrite.Drawing:
    """
    Symmetric difference fill: by default keep regions in an *odd* number of input shapes (XOR).
    Set keep_even=True to keep regions in an *even* number (0, 2, 4...) instead.
    Uses Shapely's symmetric_difference_all when available, else sequential XOR (when keep_even=False);
    when keep_even=True uses arrangement-with-depths and keeps depth % 2 == 0.
    Either pass a list of Shapely Polygons (e.g. grid.get_shapely_polygons())
    or leave polygons=None to extract from the drawing. Modifies the drawing in place.
    """
    start_time = time.perf_counter()

    t0 = time.perf_counter()
    if polygons is not None:
        points_lists = _shapely_polygons_to_points_lists(polygons)
    else:
        points_lists = _polygon_points_from_drawing(drawing)
    print(f"[timer] symmetric_difference — get points_lists: {time.perf_counter() - t0:.4f} s")

    if not points_lists:
        return drawing

    t0 = time.perf_counter()
    shapely_polys = _points_lists_to_shapely_polys(points_lists)
    print(f"[timer] symmetric_difference — points_lists → Shapely polys: {time.perf_counter() - t0:.4f} s")

    if not shapely_polys:
        return drawing

    t0 = time.perf_counter()
    if keep_even:
        regions_list, depths_list = _build_arrangement_with_depths(points_lists)
        if regions_list is None or depths_list is None:
            return drawing
        regions = [
            region for region, depth in zip(regions_list, depths_list)
            if depth % 2 == 0 and not region.is_empty and region.area > _MIN_REGION_AREA
        ]
    else:
        shapely_polys = _ensure_valid_polygons(shapely_polys)
        if not shapely_polys:
            return drawing
        result = _xor_shapely_polygons(shapely_polys, parallel=False)
        if result is None or (hasattr(result, "is_empty") and result.is_empty):
            return drawing
        regions = [
            region for region in _to_polygon_list(result)
            if not region.is_empty and region.area > _MIN_REGION_AREA
        ]
    print(f"[timer] symmetric_difference — regions (keep_even={keep_even}): {time.perf_counter() - t0:.4f} s")

    t0 = time.perf_counter()
    _draw_regions(drawing, regions)
    print(f"[timer] symmetric_difference — _draw_regions: {time.perf_counter() - t0:.4f} s")

    print(f"[timer] symmetric_difference (total): {time.perf_counter() - start_time:.4f} s")
    return drawing


def symmetric_difference_parallel(
    drawing: svgwrite.Drawing, polygons=None, keep_even: bool = False
) -> svgwrite.Drawing:
    """
    XOR of all input polygons; result is drawn (odd-coverage regions).
    keep_even=True: prepend a large bounding square so XOR gives even-coverage regions instead.
    Uses Shapely symmetric_difference_all when available, else parallel iterative XOR.
    """
    start_time = time.perf_counter()

    t0 = time.perf_counter()
    if polygons is not None:
        points_lists = _shapely_polygons_to_points_lists(polygons)
    else:
        points_lists = _polygon_points_from_drawing(drawing)
    print(f"[timer] symmetric_difference_parallel — get points_lists: {time.perf_counter() - t0:.4f} s")

    if not points_lists:
        return drawing

    t0 = time.perf_counter()
    shapely_polys = _points_lists_to_shapely_polys(points_lists)
    print(f"[timer] symmetric_difference_parallel — points_lists → Shapely polys: {time.perf_counter() - t0:.4f} s")

    if not shapely_polys:
        return drawing

    # When keep_even, add one big square around everything so XOR yields even-coverage regions.
    if keep_even:
        union_all = unary_union(shapely_polys)
        minx, miny, maxx, maxy = union_all.bounds
        pad = max(maxx - minx, maxy - miny) * 0.1 or 1.0
        wrapper = ShapelyPolygon([
            (minx - pad, miny - pad), (maxx + pad, miny - pad),
            (maxx + pad, maxy + pad), (minx - pad, maxy + pad),
        ])
        shapely_polys = list(shapely_polys) + [wrapper]

    # GEOS can raise TopologyException on invalid or sliver geometry; normalize before set op.
    shapely_polys = _ensure_valid_polygons(shapely_polys)
    if not shapely_polys:
        return drawing

    t0 = time.perf_counter()
    result = _xor_shapely_polygons(shapely_polys, parallel=True)
    if result is None or (hasattr(result, "is_empty") and result.is_empty):
        return drawing
    regions = [
        r for r in _to_polygon_list(result)
        if not r.is_empty and r.area > _MIN_REGION_AREA
    ]
    print(f"[timer] symmetric_difference_parallel — regions (keep_even={keep_even}): {time.perf_counter() - t0:.4f} s")

    t0 = time.perf_counter()
    _draw_regions(drawing, regions)
    print(f"[timer] symmetric_difference_parallel — _draw_regions: {time.perf_counter() - t0:.4f} s")

    print(f"[timer] symmetric_difference_parallel (total): {time.perf_counter() - start_time:.4f} s")
    return drawing