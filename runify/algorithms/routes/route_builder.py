"""
Old and dosent work right now, soon to be fixed
"""

import json
import os
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional

# existing grade adjustment functions
def minetti_grade_adjustment(grade_percent):
    """Minetti formula for metabolic cost adjustment based on grade"""
    i = grade_percent / 100
    cost_factor = 155.4 * (i**5) - 30.4 * (i**4) - 43.3 * (i**3) + 46.3 * (i**2) - 165 * i + 3.6
    return cost_factor / 100

def strava_grade_adjustment(grade_percent):
    """Strava gets robbed"""
    i = grade_percent / 100
    relative_cost = 15.14 * (i**2) - 2.896 * i
    return 1 + (relative_cost / 100)

def simple_grade_adjustment(grade_percent):
    """Simple linear approximation: 10% grade = 30% harder"""
    return 1 + (abs(grade_percent) * 0.03)

# Helper functions
def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points in meters"""
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371000  # Earth radius in meters

def compute_grade(elev1, elev2, dist_meters):
    """Calculate grade percentage between two points"""
    if dist_meters == 0:
        return 0
    return ((elev2 - elev1) / dist_meters) * 100

class RouteBuilder:
    def __init__(self, trail_data, grade_model=minetti_grade_adjustment):
        """Initialize route builder with trail data and grade model"""
        self.trail_data = trail_data
        self.grade_model = grade_model
        self.trail_graph = self._build_trail_graph()
    
    def _build_trail_graph(self):
        """Build connected graph of all trail points"""
        graph = {}
        point_id = 0
        
        for trail in self.trail_data:
            trail_points = trail.get('latlng', [])
            if len(trail_points) < 2:
                continue
                
            trail_point_ids = []
            for point in trail_points:
                graph[point_id] = []
                trail_point_ids.append((point_id, point))
                point_id += 1
            
            # Connect sequential points in trail
            for i in range(len(trail_point_ids) - 1):
                id1, pt1 = trail_point_ids[i]
                id2, pt2 = trail_point_ids[i + 1]
                
                dist = haversine_distance(pt1[0], pt1[1], pt2[0], pt2[1])
                grade = 0
                if len(pt1) > 2 and len(pt2) > 2:
                    grade = compute_grade(pt1[2], pt2[2], dist)
                
                effort = dist * self.grade_model(grade)
                
                # Bidirectional connections
                segment_info = {
                    'distance': dist,
                    'grade': grade, 
                    'effort': effort,
                    'point': pt2
                }
                graph[id1].append((id2, segment_info))
                
                reverse_segment = {
                    'distance': dist,
                    'grade': -grade,
                    'effort': dist * self.grade_model(-grade),
                    'point': pt1
                }
                graph[id2].append((id1, reverse_segment))
        
        return graph
    
    def find_routes_by_effort(self, start_lat, start_lng, target_distance_miles, 
                             target_pace_min_mile, tolerance=0.15):
        """Find routes matching target distance with balanced effort"""
        target_distance_m = target_distance_miles * 1609.34
        
        start_point_id = self._find_nearest_point(start_lat, start_lng)
        if start_point_id is None:
            return []
        
        print(f"Building routes for {target_distance_miles} miles at {target_pace_min_mile} min/mile pace")
        
        # Generate candidate routes
        candidate_routes = self._generate_route_candidates(
            start_point_id, target_distance_m, tolerance
        )
        
        # Evaluate and rank routes
        evaluated_routes = []
        for route in candidate_routes:
            route_analysis = self._analyze_route_effort(route)
            if route_analysis:
                evaluated_routes.append(route_analysis)
        
        # Sort by effort score (lower is better)
        evaluated_routes.sort(key=lambda x: x['effort_score'])
        return evaluated_routes[:5]
    
    def _find_nearest_point(self, lat, lng):
        """Find nearest trail point to coordinates"""
        min_dist = float('inf')
        nearest_id = None
        
        for point_id, connections in self.trail_graph.items():
            if not connections:
                continue
            point = connections[0][1]['point']
            if len(point) >= 2:
                dist = haversine_distance(lat, lng, point[0], point[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = point_id
        
        return nearest_id
    
    def _generate_route_candidates(self, start_id, target_distance, tolerance):
        """Generate multiple route candidates"""
        candidates = []
        max_distance = target_distance * (1 + tolerance)
        min_distance = target_distance * (1 - tolerance)
        
        # Random walks
        for _ in range(50):
            route = self._random_walk(start_id, min_distance, max_distance)
            if route and min_distance <= self._route_distance(route) <= max_distance:
                candidates.append(route)
        
        # Loop-seeking walks
        for _ in range(30):
            route = self._loop_walk(start_id, min_distance, max_distance)
            if route and min_distance <= self._route_distance(route) <= max_distance:
                candidates.append(route)
        
        return candidates
    
    def _random_walk(self, start_id, min_dist, max_dist):
        """Generate route by random walking through trail network"""
        route = [start_id]
        current_id = start_id
        total_distance = 0
        max_steps = 200
        
        for step in range(max_steps):
            if current_id not in self.trail_graph:
                break
                
            connections = self.trail_graph[current_id]
            if not connections:
                break
            
            # Prefer unvisited points
            unvisited = [(next_id, info) for next_id, info in connections 
                        if next_id not in route[-20:]]
            
            if unvisited:
                next_id, segment_info = random.choice(unvisited)
            else:
                next_id, segment_info = random.choice(connections)
            
            if total_distance + segment_info['distance'] > max_dist:
                if total_distance >= min_dist:
                    break
                else:
                    continue
            
            route.append(next_id)
            total_distance += segment_info['distance']
            current_id = next_id
            
            if total_distance >= min_dist and random.random() < 0.3:
                break
        
        return route if len(route) > 1 else None
    
    def _loop_walk(self, start_id, min_dist, max_dist):
        """Generate route that tries to loop back near start"""
        route = self._random_walk(start_id, min_dist * 0.8, max_dist * 0.9)
        if not route:
            return None
        
        # Try to find path back towards start
        current_id = route[-1]
        start_connections = self.trail_graph.get(start_id, [])
        if not start_connections:
            return route
        
        start_point = start_connections[0][1]['point'] if start_connections else None
        if not start_point:
            return route
        
        # Look for points near start area
        for next_id, segment_info in self.trail_graph.get(current_id, []):
            next_point = segment_info['point']
            if len(next_point) >= 2 and len(start_point) >= 2:
                dist_to_start = haversine_distance(
                    next_point[0], next_point[1], start_point[0], start_point[1]
                )
                if dist_to_start < 500:  # Within 500m of start
                    route.append(next_id)
                    break
        
        return route
    
    def _route_distance(self, route):
        """Calculate total distance of route"""
        if len(route) < 2:
            return 0
        
        total = 0
        for i in range(len(route) - 1):
            current_id = route[i]
            next_id = route[i + 1]
            
            for conn_id, segment_info in self.trail_graph.get(current_id, []):
                if conn_id == next_id:
                    total += segment_info['distance']
                    break
        
        return total
    
    def _analyze_route_effort(self, route):
        """Analyze effort distribution and score route"""
        if len(route) < 2:
            return None
        
        segments = []
        total_distance = 0
        total_effort = 0
        elevations = []
        
        for i in range(len(route) - 1):
            current_id = route[i]
            next_id = route[i + 1]
            
            segment_info = None
            for conn_id, info in self.trail_graph.get(current_id, []):
                if conn_id == next_id:
                    segment_info = info
                    break
            
            if segment_info:
                segments.append(segment_info)
                total_distance += segment_info['distance']
                total_effort += segment_info['effort']
                
                point = segment_info['point']
                if len(point) > 2:
                    elevations.append(point[2])
        
        if not segments:
            return None
        
        # Calculate effort score
        efforts = [seg['effort'] for seg in segments]
        effort_variance = np.var(efforts) if len(efforts) > 1 else 0
        elev_variation = np.std(elevations) if len(elevations) > 1 else 0
        effort_score = effort_variance - (elev_variation * 0.1)
        
        return {
            'route': route,
            'segments': segments,
            'total_distance': total_distance,
            'total_effort': total_effort,
            'effort_score': effort_score,
            'distance_miles': total_distance / 1609.34,
            'avg_effort_per_mile': total_effort / (total_distance / 1609.34) if total_distance > 0 else 0,
            'elevation_profile': elevations,
            'route_points': self._get_route_coordinates(route)
        }
    
    def _get_route_coordinates(self, route):
        """Convert route IDs to lat/lng coordinates"""
        coordinates = []
        for i, point_id in enumerate(route):
            connections = self.trail_graph.get(point_id, [])
            if connections:
                point = connections[0][1]['point']
                coordinates.append(point)
            elif i > 0:  # Use previous point's target
                prev_id = route[i-1]
                for conn_id, info in self.trail_graph.get(prev_id, []):
                    if conn_id == point_id:
                        coordinates.append(info['point'])
                        break
        return coordinates

def load_trail_database(trails_db_folder):
    """Load all trails from trails_db folder"""
    trails = []
    
    # slop that dosent work too well rn
    way_folder = os.path.join(trails_db_folder)
    print(way_folder)
    if os.path.exists(way_folder):
        for fname in os.listdir(way_folder):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(way_folder, fname), 'r') as f:
                        data = json.load(f)
                        if 'latlng' in data and 'data' in data['latlng']:
                            latlng = data['latlng']['data']
                            # Try to merge elevation samples if available
                            elevs = data.get('elevation_stats', {}).get('samples', [])
                            if elevs and len(elevs) == len(latlng):
                                latlng = [latlng[i] + [elevs[i]] for i in range(len(latlng))]
                            trails.append({'trail_id': fname[:-5], 'latlng': latlng, 'source': 'osm_derived'})
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

    # custom stuff yay
    if os.path.exists(trails_db_folder):
        for fname in os.listdir(trails_db_folder):
            if fname.endswith(('_raw.json', '_custom.json')):
                try:
                    with open(os.path.join(trails_db_folder, fname), 'r') as f:
                        data = json.load(f)
                        if 'latlng' in data and 'data' in data['latlng']:
                            trails.append({
                                'trail_id': fname[:-5],
                                'latlng': data['latlng']['data'],
                                'source': 'user_generated',
                                'elevation_stats': data.get('elevation_stats', {})
                            })
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
    
    return trails

def create_osm_fallback_routes(start_lat, start_lng, target_distance_miles, target_pace):
    """Simple OSM fallback - creates circular route estimate"""
    target_distance_m = target_distance_miles * 1609.34
    route_radius = target_distance_m / (2 * math.pi)
    route_points = []
    
    # Generate circular route
    num_points = 20
    for i in range(num_points + 1):
        angle = (i / num_points) * 2 * math.pi
        lat = start_lat + (route_radius / 111111) * math.cos(angle)
        lng = start_lng + (route_radius / (111111 * math.cos(math.radians(start_lat)))) * math.sin(angle)
        route_points.append([lat, lng])
    
    total_distance = sum(haversine_distance(route_points[i][0], route_points[i][1], 
                                          route_points[i+1][0], route_points[i+1][1])
                        for i in range(len(route_points)-1))
    
    return [{
        'route_points': route_points,
        'total_distance': total_distance,
        'distance_miles': total_distance / 1609.34,
        'total_effort': total_distance,
        'source': 'osm_fallback',
        'has_elevation': False
    }]

def generate_workout_route(start_lat, start_lng, distance_miles, pace_min_mile, 
                          effort_preference="balanced", grade_model_name="minetti",
                          trails_db_folder="trails_db/way"):
    """
    Main function to generate workout route
    
    Args:
        start_lat, start_lng: Starting coordinates
        distance_miles: Target distance in miles
        pace_min_mile: Target pace in minutes per mile
        effort_preference: "balanced", "harder_downhills", "easier_uphills"
        grade_model_name: "minetti", "strava", "simple"
        trails_db_folder: Path to trails database
    
    Returns:
        Dict with routes and metadata
    """
    
    # Select grade model
    grade_models = {
        "minetti": minetti_grade_adjustment,
        "strava": strava_grade_adjustment,
        "simple": simple_grade_adjustment
    }
    grade_model = grade_models.get(grade_model_name, minetti_grade_adjustment)
    orig_grade_model = grade_model
    def modified_grade_model(grade):
        base_effort = orig_grade_model(grade)
        if grade < 0:
            return base_effort * 1.5
        return base_effort
    grade_model = modified_grade_model

    
    # Modify based on effort preference
    if effort_preference == "harder_downhills":
        def modified_grade_model(grade):
            base_effort = grade_model(grade)
            return base_effort * 1.5 if grade < 0 else base_effort
        grade_model = modified_grade_model
        
    elif effort_preference == "easier_uphills":
        def modified_grade_model(grade):
            base_effort = grade_model(grade)
            return base_effort * 0.7 if grade > 0 else base_effort
        grade_model = modified_grade_model
    
    # Load trail database
    trails = load_trail_database(trails_db_folder)
    print(f"Loaded {len(trails)} trails from database")
    
    trails_with_elevation = [t for t in trails if any(len(pt) > 2 for pt in t['latlng'])]
    print(f"  {len(trails_with_elevation)} trails have elevation data")
    
    # Try to build route with existing data
    if len(trails) >= 2:
        builder = RouteBuilder(trails, grade_model)
        routes = builder.find_routes_by_effort(start_lat, start_lng, 
                                             distance_miles, pace_min_mile)
        
        if routes:
            print(f"Successfully built {len(routes)} routes using trail database")
            return {
                'routes': routes,
                'source': 'trail_database',
                'parameters': {
                    'distance_miles': distance_miles,
                    'pace_min_mile': pace_min_mile,
                    'effort_preference': effort_preference,
                    'grade_model': grade_model_name
                }
            }
    
    # OSM fallback - need to fix
    print("Insufficient trail data, falling back to OSM routing")
    osm_routes = create_osm_fallback_routes(start_lat, start_lng, distance_miles, pace_min_mile)
    
    return {
        'routes': osm_routes,
        'source': 'osm_fallback',
        'parameters': {
            'distance_miles': distance_miles,
            'pace_min_mile': pace_min_mile,
            'effort_preference': effort_preference,
            'grade_model': grade_model_name
        }
    }

# Example (testing)
if __name__ == "__main__":
    # Generate a 5-mile route at 8 min/mile pace with harder downhills
    result = generate_workout_route(
        start_lat=35.786961,
        start_lng=-78.750545,
        distance_miles=0.1,
        pace_min_mile=8.0,
        effort_preference="harder_downhills",
        grade_model_name="minetti",
        trails_db_folder="trails_db/way"
    )
    
    print(f"Generated {len(result['routes'])} routes using {result['source']}")
    if result['routes']:
        best_route = result['routes'][0]
        print(f"Best route: {best_route.get('distance_miles', 0):.2f} miles")
