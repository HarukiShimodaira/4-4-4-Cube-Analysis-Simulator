"""
Edge Tracker Module
Identifies and tracks all 24 edge pieces (wing cubes) for the 4×4×4 Cube Analysis Simulator

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from ..core.cube4x4 import Cube4x4
from ..core.vector3 import Vector3


@dataclass
class EdgeInfo:
    """Information about a single edge piece (wing cube)"""
    edge_id: str  # e.g., "UF_1", "UF_2"
    pair_id: str  # e.g., "UF" (the pair this edge belongs to)
    index: int  # 0 or 1 (which piece in the pair)
    initial_position: Vector3
    current_position: Vector3
    cubelet_id: int  # Reference to the actual cubelet
    
    @property
    def moved_distance(self) -> float:
        """Calculate Euclidean distance from initial position"""
        dx = self.current_position.x - self.initial_position.x
        dy = self.current_position.y - self.initial_position.y
        dz = self.current_position.z - self.initial_position.z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Round to common exact values
        return EdgeTracker.round_distance(dist)


class EdgeTracker:
    """
    Tracks all 24 edge pieces in a 4×4×4 cube
    
    Edge naming convention:
    - 12 pairs: UF, UR, UB, UL, DF, DR, DB, DL, FR, FL, BR, BL
    
    Coordinate system:
    - Outer layers: ±1.5
    - Inner layers: ±0.5
    """
    
    # Tolerance for floating point comparisons
    EPSILON = 1e-9
    
    # Valid coordinate values in 4×4×4 cube
    VALID_COORDS = [-1.5, -0.5, 0.5, 1.5]
    
    @staticmethod
    def round_coord(value: float) -> float:
        """
        Round coordinate to nearest valid value, accounting for floating point errors
        Valid values: -1.5, -0.5, 0.5, 1.5
        """
        # Find nearest valid coordinate
        nearest = min(EdgeTracker.VALID_COORDS, key=lambda v: abs(v - value))
        # If within tolerance, return the exact valid value
        if abs(value - nearest) < EdgeTracker.EPSILON:
            return nearest
        return value
    
    @staticmethod
    def round_distance(distance: float) -> float:
        """
        Round distance to mathematically correct value
        Common distances in 4×4×4 cube:
        - 1.0 (adjacent on same edge)
        - sqrt(2) ≈ 1.414 (diagonal on same face)
        - sqrt(3) ≈ 1.732 (space diagonal between corners of unit cube)
        - sqrt(10) ≈ 3.162 (separated edge pair after inner rotation)
        """
        import math
        
        # Common exact distances in 4×4×4 cube
        common_distances = [
            (1.0, 1.0),
            (math.sqrt(2), math.sqrt(2)),
            (math.sqrt(3), math.sqrt(3)),
            (2.0, 2.0),
            (math.sqrt(5), math.sqrt(5)),
            (math.sqrt(8), math.sqrt(8)),
            (3.0, 3.0),
            (math.sqrt(10), math.sqrt(10)),
        ]
        
        # Find if distance matches a known exact value
        for exact_val, _ in common_distances:
            if abs(distance - exact_val) < EdgeTracker.EPSILON:
                return exact_val
        
        return distance
    
    # Define the 12 edge pairs and their initial positions
    # Each pair consists of two edges that are on the same edge in a 3x3x3 cube
    # They share 2 coordinates and differ in one coordinate (one at ±1.5, one at ±0.5)
    
    # Define the 12 edge pairs and their initial positions
    # Each pair consists of two edges that are on the same edge in a 3x3x3 cube
    # They share 2 coordinates and differ in one coordinate (one at ±1.5, one at ±0.5)
    EDGE_PAIRS = {
        # U face edges (y=1.5)
        'UF': [(0.5, 1.5, 1.5), (-0.5, 1.5, 1.5)],  # Up-Front: x differs
        'UR': [(1.5, 1.5, 0.5), (1.5, 1.5, -0.5)],  # Up-Right: z differs
        'UB': [(0.5, 1.5, -1.5), (-0.5, 1.5, -1.5)],  # Up-Back: x differs
        'UL': [(-1.5, 1.5, 0.5), (-1.5, 1.5, -0.5)],  # Up-Left: z differs
        # D face edges (y=-1.5)
        'DF': [(0.5, -1.5, 1.5), (-0.5, -1.5, 1.5)],  # Down-Front: x differs
        'DR': [(1.5, -1.5, 0.5), (1.5, -1.5, -0.5)],  # Down-Right: z differs
        'DB': [(0.5, -1.5, -1.5), (-0.5, -1.5, -1.5)],  # Down-Back: x differs
        'DL': [(-1.5, -1.5, 0.5), (-1.5, -1.5, -0.5)],  # Down-Left: z differs
        # Middle edges (y=±0.5)
        'FR': [(1.5, 0.5, 1.5), (1.5, -0.5, 1.5)],  # Front-Right: y differs
        'FL': [(-1.5, 0.5, 1.5), (-1.5, -0.5, 1.5)],  # Front-Left: y differs
        'BR': [(1.5, 0.5, -1.5), (1.5, -0.5, -1.5)],  # Back-Right: y differs
        'BL': [(-1.5, 0.5, -1.5), (-1.5, -0.5, -1.5)],  # Back-Left: y differs
    }
    
    def __init__(self):
        """Initialize the edge tracker"""
        self.initial_edges: Dict[str, EdgeInfo] = {}
        self._initialize_edge_positions()
    
    def _initialize_edge_positions(self):
        """Create initial edge information for all 24 edges"""
        for pair_id, positions in self.EDGE_PAIRS.items():
            for idx, pos in enumerate(positions):
                edge_id = f"{pair_id}_{idx + 1}"
                self.initial_edges[edge_id] = EdgeInfo(
                    edge_id=edge_id,
                    pair_id=pair_id,
                    index=idx,
                    initial_position=Vector3(pos[0], pos[1], pos[2]),
                    current_position=Vector3(pos[0], pos[1], pos[2]),
                    cubelet_id=-1  # Will be set when tracking actual cube
                )
    
    def identify_edges(self, cube: Cube4x4) -> List[EdgeInfo]:
        """
        Identify all 24 edge pieces in the current cube state
        
        Args:
            cube: The cube to analyze
            
        Returns:
            List of EdgeInfo objects with current positions
            
        Raises:
            ValueError: If edge count is not exactly 24
        """
        edges = []
        
        # Find all edge cubelets (type == "edge")
        edge_cubelets = [c for c in cube.cubelets if c.type == "edge"]
        
        # Validate edge count
        if len(edge_cubelets) != 24:
            raise ValueError(f"Invalid cube state: found {len(edge_cubelets)} edges, expected 24")
        
        # For each edge cubelet, identify which initial edge it is
        for cubelet in edge_cubelets:
            # Find which initial edge this cubelet corresponds to
            # by matching its initial_position
            initial_pos = cubelet.initial_position
            
            # Find the edge ID for this initial position
            edge_id = None
            for eid, edge_info in self.initial_edges.items():
                init_edge_pos = edge_info.initial_position
                if (abs(init_edge_pos.x - initial_pos.x) < 1e-6 and
                    abs(init_edge_pos.y - initial_pos.y) < 1e-6 and
                    abs(init_edge_pos.z - initial_pos.z) < 1e-6):
                    edge_id = eid
                    break
            
            if edge_id:
                initial_edge = self.initial_edges[edge_id]
                # Round coordinates to account for floating point errors
                rounded_pos = Vector3(
                    self.round_coord(cubelet.position.x),
                    self.round_coord(cubelet.position.y),
                    self.round_coord(cubelet.position.z)
                )
                current_edge = EdgeInfo(
                    edge_id=initial_edge.edge_id,
                    pair_id=initial_edge.pair_id,
                    index=initial_edge.index,
                    initial_position=initial_edge.initial_position,
                    current_position=rounded_pos,
                    cubelet_id=cubelet.id
                )
                edges.append(current_edge)
            else:
                raise ValueError(f"Could not identify edge at initial position ({initial_pos.x}, {initial_pos.y}, {initial_pos.z})")
        
        # Final validation: ensure we have exactly 24 edges
        if len(edges) != 24:
            raise ValueError(f"Edge identification failed: found {len(edges)} edges, expected 24")
        
        return edges
    
    def _distance(self, pos1: Vector3, pos2: Vector3) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        return self.round_distance(dist)
    
    def get_edge_pairs(self, edges: List[EdgeInfo]) -> Dict[str, Tuple[EdgeInfo, EdgeInfo]]:
        """
        Group edges into their pairs
        
        Args:
            edges: List of EdgeInfo objects
            
        Returns:
            Dictionary mapping pair_id to tuple of (edge_1, edge_2)
            
        Raises:
            ValueError: If any pair doesn't have exactly 2 edges
        """
        pairs = {}
        
        for pair_id in self.EDGE_PAIRS.keys():
            pair_edges = [e for e in edges if e.pair_id == pair_id]
            if len(pair_edges) != 2:
                raise ValueError(f"Invalid pair {pair_id}: found {len(pair_edges)} edges, expected 2")
            # Sort by index to ensure consistent order
            pair_edges.sort(key=lambda e: e.index)
            pairs[pair_id] = tuple(pair_edges)
        
        # Validate we have all 12 pairs
        if len(pairs) != 12:
            raise ValueError(f"Invalid pairs: found {len(pairs)} pairs, expected 12")
        
        return pairs
    
    def calculate_pair_distance(self, edge1: EdgeInfo, edge2: EdgeInfo) -> float:
        """Calculate distance between two edges in a pair"""
        return self._distance(edge1.current_position, edge2.current_position)
    
    def check_same_face(self, edge1: EdgeInfo, edge2: EdgeInfo) -> Optional[str]:
        """
        Check if two edges are on the same edge in 3x3x3 terms
        エッジは2つの面の境界にあるため、同じ2つの面の組み合わせに属しているかチェック
        
        Returns:
            Edge position name (e.g., 'UF', 'UR') if on same edge, None otherwise
        """
        pos1 = edge1.current_position
        pos2 = edge2.current_position
        
        epsilon = 1e-6
        
        # 各エッジの面を特定（エッジは2つの面の境界にある）
        def get_faces(pos):
            faces = []
            if abs(pos.y - 1.5) < epsilon:
                faces.append('U')
            elif abs(pos.y + 1.5) < epsilon:
                faces.append('D')
            
            if abs(pos.z - 1.5) < epsilon:
                faces.append('F')
            elif abs(pos.z + 1.5) < epsilon:
                faces.append('B')
            
            if abs(pos.x - 1.5) < epsilon:
                faces.append('R')
            elif abs(pos.x + 1.5) < epsilon:
                faces.append('L')
            
            return set(faces)
        
        faces1 = get_faces(pos1)
        faces2 = get_faces(pos2)
        
        # 両方のエッジが同じ2つの面の組み合わせに属している場合のみ同一エッジ
        if faces1 == faces2 and len(faces1) == 2:
            # ソートして一貫した順序で返す
            return ''.join(sorted(faces1))
        
        return None
    
    def get_adjacent_edges(self, edge: EdgeInfo, all_edges: List[EdgeInfo]) -> List[Dict]:
        """
        Find all edges adjacent to the given edge
        
        Args:
            edge: The edge to find neighbors for
            all_edges: List of all edges in current state
            
        Returns:
            List of dicts with 'edge', 'distance', and 'shared_face' info
        """
        adjacent = []
        
        for other_edge in all_edges:
            if other_edge.edge_id == edge.edge_id:
                continue
            
            dist = self._distance(edge.current_position, other_edge.current_position)
            shared_face = self.check_same_face(edge, other_edge)
            
            # Consider adjacent if distance is small (< 2.5 units)
            if dist < 2.5:
                adjacent.append({
                    'edge': other_edge,
                    'distance': dist,
                    'shared_face': shared_face
                })
        
        # Sort by distance
        adjacent.sort(key=lambda x: x['distance'])
        
        return adjacent
