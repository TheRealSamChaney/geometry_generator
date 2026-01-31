import svgwrite
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    try:
        from shapely import symmetric_difference_all
    except ImportError:
        symmetric_difference_all = None
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    symmetric_difference_all = None

# This will be a SVG geometric pattern creator app

width = 100
height = 400
 
output_folder = os.path.dirname(os.path.abspath(__file__))
output_filename = "geometry_generator_output.svg"
output_path = os.path.join(output_folder, output_filename)
drawing_global = svgwrite.Drawing(output_path)

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
        
class EdgeMandala: # TODO figure this shit out
    # To determine orientation of line segment AB with respect to origin C, use dot product of AC and BC. I think you can also use the cross product
    # This function should have an initial seed polygon, then you can make layers from it. Each layer will be the full set of polygons connected to all outside edges from the previous layer
    def __init__(self,  seed_polygon):
        self._seed_polygon = seed_polygon
        self.layers = [[self.seed_polygon]] # Start layers list off with seed polygon as the first layer
        
    def generate_edge_polygon(self, edge, num_points):
        start = edge[0]
        end = edge[1]
        return create_edge_polygon(start, end, num_points)
    
    def generate_layer(self):
        # Create 1 "layer" of polygons which is a full set of polygons connected to all the outside edges of the previous layer starting with the seed polygon
        last_layer = self.layers[-1]
    
    @property
    def seed_polygon(self):
        return self._seed_polygon
    
    @seed_polygon.setter
    def seed_polygon(self, polygon):
        self._seed_polygon = polygon
        
    
def create_edge_polygon(start, end, num_vertices): # ChatGPT, this one doesn't know which side of the line to make the polygon on
    # Calculate the length of the line segment
    line_length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    
    # Calculate the angle between each vertex
    angle = 2 * math.pi / num_vertices
    
    # Calculate the x and y increments for each vertex
    x_increment = line_length * math.cos(angle)
    y_increment = line_length * math.sin(angle)
    
    # Calculate the slope and y-intercept of the line segment
    slope = (end[1] - start[1]) / (end[0] - start[0])
    y_intercept = start[1] - slope * start[0]
    
    # Initialize the starting x and y coordinates
    x = start[0] + line_length / 2
    y = slope * x + y_intercept
    
    # Initialize an empty list to store the vertex coordinates
    vertices = []
    
    # Iterate through each vertex, calculating its coordinates
    for i in range(num_vertices):
        vertices.append((x, y))
        # Rotate the line segment by the angle to get the coordinates of the next vertex
        x, y = (x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle))
        # Add the x and y increments to get the coordinates of the next vertex
        x += x_increment
        y += y_increment
    
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
        
def ripple_morph(grid:GridIsometric, polygon:Polygon, i:int, j:int, magnitude:float, decrease_out:bool = True):
    # TODO implement this, similar to cirlce_morph but effect waxes and wanes according to sine function
    pass
        
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
        try:
            polygon = ShapelyPolygon(vertex_list)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            # One Shapely poly might become several (e.g. after buffer(0)); flatten them.
            for polygon_piece in _to_polygon_list(polygon):
                if not polygon_piece.is_empty and polygon_piece.area > 0:
                    shapely_polygons.append(polygon_piece)
        except Exception:
            continue
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
    if not SHAPELY_AVAILABLE:
        raise ImportError("fracture requires Shapely: pip install shapely")

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


def _iterative_symmetric_difference_chunk(polygon_list):
    """
    Worker for parallel symmetric_difference: compute iterative XOR for one chunk of
    Shapely polygons. Must be module-level for ProcessPoolExecutor pickling.
    Returns a single Shapely geometry or None if polygon_list is empty.
    """
    if not polygon_list:
        return None
    result = polygon_list[0]
    for current_polygon in polygon_list[1:]:
        result = result.symmetric_difference(current_polygon)
    return result


def symmetric_difference(
    drawing: svgwrite.Drawing, polygons=None
) -> svgwrite.Drawing:
    """
    Symmetric difference (XOR) fill: keep regions in an odd number of input shapes.
    Uses Shapely's symmetric_difference_all(geometries) when available (one batch op);
    otherwise falls back to sequential iterative XOR.
    Either pass a list of Shapely Polygons (e.g. grid.get_shapely_polygons())
    or leave polygons=None to extract from the drawing. Modifies the drawing in place.
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("symmetric_difference requires Shapely: pip install shapely")

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
    if symmetric_difference_all is not None:
        result = symmetric_difference_all(shapely_polys)
    else:
        result = _iterative_symmetric_difference_chunk(shapely_polys)
    print(f"[timer] symmetric_difference — XOR (batch or sequential): {time.perf_counter() - t0:.4f} s")

    if result is None or (hasattr(result, "is_empty") and result.is_empty):
        return drawing

    t0 = time.perf_counter()
    regions = [
        region for region in _to_polygon_list(result)
        if not region.is_empty and region.area > _MIN_REGION_AREA
    ]
    print(f"[timer] symmetric_difference — result → regions list: {time.perf_counter() - t0:.4f} s")

    t0 = time.perf_counter()
    _draw_regions(drawing, regions)
    print(f"[timer] symmetric_difference — _draw_regions: {time.perf_counter() - t0:.4f} s")

    print(f"[timer] symmetric_difference (total): {time.perf_counter() - start_time:.4f} s")
    return drawing


def symmetric_difference_parallel(
    drawing: svgwrite.Drawing, polygons=None
) -> svgwrite.Drawing:
    """
    Same as symmetric_difference (XOR fill) but uses parallel iterative XOR: chunk polygons
    across CPU workers, each worker does iterative symmetric_difference on its chunk, then
    main process combines partial results. Use when symmetric_difference_all is slow or unavailable.
    Same signature as symmetric_difference; modifies the drawing in place.
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("symmetric_difference_parallel requires Shapely: pip install shapely")

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

    t0 = time.perf_counter()
    max_workers = min(os.cpu_count() or 4, len(shapely_polys))
    if max_workers <= 1:
        result = _iterative_symmetric_difference_chunk(shapely_polys)
    else:
        chunks = [
            [shapely_polys[index] for index in range(worker_index, len(shapely_polys), max_workers)]
            for worker_index in range(max_workers)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            partial_results = list(executor.map(_iterative_symmetric_difference_chunk, chunks))
        result = None
        for partial in partial_results:
            if partial is not None:
                result = partial if result is None else result.symmetric_difference(partial)
    print(f"[timer] symmetric_difference_parallel — parallel iterative XOR: {time.perf_counter() - t0:.4f} s")

    if result is None:
        return drawing

    t0 = time.perf_counter()
    regions = [
        region for region in _to_polygon_list(result)
        if not region.is_empty and region.area > _MIN_REGION_AREA
    ]
    print(f"[timer] symmetric_difference_parallel — result → regions list: {time.perf_counter() - t0:.4f} s")

    t0 = time.perf_counter()
    _draw_regions(drawing, regions)
    print(f"[timer] symmetric_difference_parallel — _draw_regions: {time.perf_counter() - t0:.4f} s")

    print(f"[timer] symmetric_difference_parallel (total): {time.perf_counter() - start_time:.4f} s")
    return drawing