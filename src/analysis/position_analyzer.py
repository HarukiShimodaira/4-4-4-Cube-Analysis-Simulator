"""
Position Analyzer Module
Analyzes positional relationships between edge pairs

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from .edge_tracker import EdgeTracker, EdgeInfo


@dataclass
class PairAnalysis:
    """Analysis of a single edge pair"""
    pair_id: str
    edge1: EdgeInfo
    edge2: EdgeInfo
    initial_distance: float  # Always 1.0 for adjacent pairs
    current_distance: float
    same_face: bool
    shared_face: str  # Face name if same_face is True
    separated: bool  # If distance > initial_distance
    distance_category: str  # 'near', 'medium', 'far'
    edge1_moved: float
    edge2_moved: float


@dataclass
class PositionStats:
    """Overall position statistics"""
    total_pairs: int
    separated_pairs: int
    pairs_on_same_face: int
    average_pair_distance: float
    max_pair_distance: float
    min_pair_distance: float
    face_distribution: Dict[str, int]  # Count of edges per face


class PositionAnalyzer:
    """
    Analyzes positional relationships and patterns in edge pieces
    """
    
    def __init__(self):
        """Initialize the position analyzer"""
        self.tracker = EdgeTracker()
    
    def analyze_all_pairs(self, edges: List[EdgeInfo]) -> List[PairAnalysis]:
        """
        Analyze all edge pairs
        
        Args:
            edges: List of EdgeInfo objects
            
        Returns:
            List of PairAnalysis objects (always 12 pairs)
            
        Raises:
            ValueError: If analysis validation fails
        """
        pairs = self.tracker.get_edge_pairs(edges)
        analyses = []
        
        for pair_id, (edge1, edge2) in pairs.items():
            current_dist = self.tracker.calculate_pair_distance(edge1, edge2)
            same_face_result = self.tracker.check_same_face(edge1, edge2)
            
            # Calculate initial distance for validation
            initial_dist = self.tracker._distance(edge1.initial_position, edge2.initial_position)
            
            # Validate initial distance (should be 1.0 for all pairs)
            if abs(initial_dist - 1.0) > 0.01:
                raise ValueError(f"Invalid initial distance for pair {pair_id}: {initial_dist:.2f}, expected 1.0")
            
            # Categorize distance (統一基準)
            if current_dist <= 1.5:
                category = 'near'
            elif current_dist <= 3.5:
                category = 'medium'
            else:
                category = 'far'
            
            analysis = PairAnalysis(
                pair_id=pair_id,
                edge1=edge1,
                edge2=edge2,
                initial_distance=initial_dist,
                current_distance=current_dist,
                same_face=same_face_result is not None,
                shared_face=same_face_result if same_face_result else '',
                separated=current_dist > 1.1,  # Small tolerance
                distance_category=category,
                edge1_moved=edge1.moved_distance,
                edge2_moved=edge2.moved_distance
            )
            analyses.append(analysis)
        
        # Validate we have exactly 12 analyses
        if len(analyses) != 12:
            raise ValueError(f"Analysis failed: found {len(analyses)} pairs, expected 12")
        
        return analyses
    
    def calculate_position_stats(self, edges: List[EdgeInfo], 
                                 pair_analyses: List[PairAnalysis]) -> PositionStats:
        """
        Calculate overall statistics
        
        Args:
            edges: List of EdgeInfo objects
            pair_analyses: List of PairAnalysis objects
            
        Returns:
            PositionStats object
        """
        # Count separated pairs
        separated_count = sum(1 for p in pair_analyses if p.separated)
        same_face_count = sum(1 for p in pair_analyses if p.same_face)
        
        # Distance statistics
        distances = [p.current_distance for p in pair_analyses]
        avg_dist = np.mean(distances) if distances else 0.0
        max_dist = max(distances) if distances else 0.0
        min_dist = min(distances) if distances else 0.0
        
        # Face distribution
        face_dist = self._calculate_face_distribution(edges)
        
        return PositionStats(
            total_pairs=len(pair_analyses),
            separated_pairs=separated_count,
            pairs_on_same_face=same_face_count,
            average_pair_distance=avg_dist,
            max_pair_distance=max_dist,
            min_pair_distance=min_dist,
            face_distribution=face_dist
        )
    
    def _calculate_face_distribution(self, edges: List[EdgeInfo]) -> Dict[str, int]:
        """
        Calculate how many edges are on each face
        
        Note: Each edge piece touches exactly 2 faces, so the sum of all
        face counts will be 48 (24 edges × 2 faces each), not 24.
        This is correct behavior.
        """
        distribution = {'U': 0, 'D': 0, 'F': 0, 'B': 0, 'R': 0, 'L': 0}
        
        for edge in edges:
            pos = edge.current_position
            faces_touched = 0
            
            # Check which faces this edge touches
            if abs(pos.y - 1.5) < 1e-6:
                distribution['U'] += 1
                faces_touched += 1
            if abs(pos.y + 1.5) < 1e-6:
                distribution['D'] += 1
                faces_touched += 1
            if abs(pos.z - 1.5) < 1e-6:
                distribution['F'] += 1
                faces_touched += 1
            if abs(pos.z + 1.5) < 1e-6:
                distribution['B'] += 1
                faces_touched += 1
            if abs(pos.x - 1.5) < 1e-6:
                distribution['R'] += 1
                faces_touched += 1
            if abs(pos.x + 1.5) < 1e-6:
                distribution['L'] += 1
                faces_touched += 1
            
            # Validate: each edge must touch exactly 2 faces
            if faces_touched != 2:
                raise ValueError(f"Edge {edge.edge_id} at ({pos.x}, {pos.y}, {pos.z}) touches {faces_touched} faces, expected 2")
        
        # Validate: total should be 48 (24 edges × 2 faces)
        total = sum(distribution.values())
        if total != 48:
            raise ValueError(f"Face distribution sum is {total}, expected 48 (24 edges × 2 faces)")
        
        return distribution
    
    def find_patterns(self, pair_analyses: List[PairAnalysis]) -> Dict:
        """
        Detect patterns in pair distributions
        
        Returns:
            Dictionary with pattern information
        """
        patterns = {
            'distance_distribution': {
                'near': 0,
                'medium': 0,
                'far': 0
            },
            'separation_rate': 0.0,
            'same_face_rate': 0.0,
            'highly_separated_pairs': [],  # Pairs with distance > 4.0
        }
        
        # Count by category
        for p in pair_analyses:
            patterns['distance_distribution'][p.distance_category] += 1
            if p.current_distance > 4.0:
                patterns['highly_separated_pairs'].append(p.pair_id)
        
        # Calculate rates
        total = len(pair_analyses)
        if total > 0:
            patterns['separation_rate'] = sum(1 for p in pair_analyses if p.separated) / total
            patterns['same_face_rate'] = sum(1 for p in pair_analyses if p.same_face) / total
        
        return patterns
