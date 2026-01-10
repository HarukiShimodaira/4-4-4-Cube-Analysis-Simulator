"""
Random Data Collector Module
Collects cubelet data through random operations for mathematical analysis

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import random
import numpy as np
from ..core.cube4x4 import Cube4x4
from ..core.cubelet import Cubelet


@dataclass
class SnapshotData:
    """Single snapshot of all cubelets at a specific operation step"""
    step: int  # 0 = initial, 1+ = after operation N
    operation: str  # Operation performed to reach this state (empty for initial)
    cubelets_data: List[Dict[str, Any]]  # Data for all 56 cubelets


class RandomDataCollector:
    """
    Collects cubelet data through random operations
    
    Operations used: R, Rp, L, Lp, U, Up, D, Dp, F, Fp, B, Bp, r, rp, l, lp, u, up, d, dp, f, fp, b, bp
    (Single layer 90-degree rotations only, no 180-degree or multi-layer rotations)
    """
    
    # Define all single-layer 90-degree operations
    OPERATIONS = [
        'R', 'Rp', 'L', 'Lp', 'U', 'Up', 'D', 'Dp', 'F', 'Fp', 'B', 'Bp',  # Outer layers
        'r', 'rp', 'l', 'lp', 'u', 'up', 'd', 'dp', 'f', 'fp', 'b', 'bp'   # Inner layers
    ]
    
    def __init__(self, seed: Optional[int] = None, allowed_operations: Optional[List[str]] = None):
        """
        Initialize the data collector
        
        Args:
            seed: Random seed for reproducibility (None = random seed)
            allowed_operations: List of allowed operations to sample from (None = all operations)
        """
        self.seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(self.seed)

        if allowed_operations is None:
            self.allowed_operations = list(self.OPERATIONS)
        else:
            if not isinstance(allowed_operations, list):
                allowed_operations = list(allowed_operations)
            if len(allowed_operations) == 0:
                raise ValueError('allowed_operations must contain at least one operation')
            unknown = sorted({op for op in allowed_operations if op not in self.OPERATIONS})
            if unknown:
                raise ValueError(f'Unknown operations: {unknown}')
            # Normalize order to match OPERATIONS and de-duplicate
            allowed_set = set(allowed_operations)
            self.allowed_operations = [op for op in self.OPERATIONS if op in allowed_set]
            if len(self.allowed_operations) == 0:
                raise ValueError('allowed_operations must contain at least one valid operation')
        
        self.snapshots: List[SnapshotData] = []
        self.operation_sequence: List[str] = []
        
    def _get_operation_method(self, cube: Cube4x4, operation: str) -> Callable:
        """Get the method for a given operation string"""
        return getattr(cube, operation)
    
    def _capture_cubelet_data(self, cubelet: Cubelet) -> Dict[str, Any]:
        """
        Capture current state data for a single cubelet
        
        Returns dynamic data only (position and rotation)
        """
        # Calculate rotation angle and axis from initial state
        # R_relative = R_current * R_initial^T
        rot_relative_matrix = cubelet.rotation.m @ cubelet.initial_rotation.transpose().m
        
        # Extract rotation angle using trace
        trace = rot_relative_matrix[0, 0] + rot_relative_matrix[1, 1] + rot_relative_matrix[2, 2]
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle_rad)
        
        # Extract rotation axis (Rodrigues' formula)
        if angle_deg < 1e-6:
            axis = [0, 0, 0]
        else:
            axis = [
                rot_relative_matrix[2, 1] - rot_relative_matrix[1, 2],
                rot_relative_matrix[0, 2] - rot_relative_matrix[2, 0],
                rot_relative_matrix[1, 0] - rot_relative_matrix[0, 1]
            ]
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = [a / axis_norm for a in axis]
            else:
                axis = [0, 0, 0]
        
        # Calculate moved distance
        dx = cubelet.position.x - cubelet.initial_position.x
        dy = cubelet.position.y - cubelet.initial_position.y
        dz = cubelet.position.z - cubelet.initial_position.z
        moved_distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        
        return {
            'id': cubelet.id,
            'pos_x': cubelet.position.x,
            'pos_y': cubelet.position.y,
            'pos_z': cubelet.position.z,
            'rot_m00': cubelet.rotation.m[0, 0],
            'rot_m01': cubelet.rotation.m[0, 1],
            'rot_m02': cubelet.rotation.m[0, 2],
            'rot_m10': cubelet.rotation.m[1, 0],
            'rot_m11': cubelet.rotation.m[1, 1],
            'rot_m12': cubelet.rotation.m[1, 2],
            'rot_m20': cubelet.rotation.m[2, 0],
            'rot_m21': cubelet.rotation.m[2, 1],
            'rot_m22': cubelet.rotation.m[2, 2],
            'rotation_angle': angle_deg,
            'rotation_axis_x': axis[0],
            'rotation_axis_y': axis[1],
            'rotation_axis_z': axis[2],
            'moved_distance': moved_distance
        }
    
    def _capture_snapshot(self, cube: Cube4x4, step: int, operation: str = "") -> SnapshotData:
        """Capture current state of all cubelets"""
        cubelets_data = [self._capture_cubelet_data(c) for c in cube.cubelets]
        return SnapshotData(
            step=step,
            operation=operation,
            cubelets_data=cubelets_data
        )
    
    def collect_data(self, num_operations: int, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
        """
        Perform random operations and collect data at each step
        
        Args:
            num_operations: Number of random operations to perform
            progress_callback: Optional callback(current, total, message) for progress updates
            
        Returns:
            Dictionary containing:
                - seed: Random seed used
                - operation_sequence: List of operations performed
                - snapshots: List of SnapshotData objects
                - initial_state: Static data for all cubelets
        """
        cube = Cube4x4()
        self.snapshots = []
        self.operation_sequence = []
        
        # Capture initial state with static data
        initial_cubelets = []
        for cubelet in cube.cubelets:
            initial_cubelets.append({
                'id': cubelet.id,
                'type': cubelet.type,
                'initial_pos_x': cubelet.initial_position.x,
                'initial_pos_y': cubelet.initial_position.y,
                'initial_pos_z': cubelet.initial_position.z
            })
        
        # Capture initial snapshot (step 0)
        if progress_callback:
            progress_callback(0, num_operations, "初期状態を記録中...")
        self.snapshots.append(self._capture_snapshot(cube, 0, ""))
        
        # Perform random operations
        for i in range(num_operations):
            # Select random operation
            operation = random.choice(self.allowed_operations)
            self.operation_sequence.append(operation)
            
            # Execute operation
            operation_method = self._get_operation_method(cube, operation)
            operation_method()
            
            # Capture snapshot
            if progress_callback:
                progress_callback(i + 1, num_operations, f"操作 {i+1}/{num_operations}: {operation}")
            self.snapshots.append(self._capture_snapshot(cube, i + 1, operation))
        
        return {
            'seed': self.seed,
            'operation_sequence': self.operation_sequence,
            'snapshots': self.snapshots,
            'initial_state': initial_cubelets
        }
    
    def get_static_data(self) -> List[Dict[str, Any]]:
        """Get static cubelet data (ID, type, initial position)"""
        if not self.snapshots:
            return []
        
        # Extract from first snapshot
        first_snapshot = self.snapshots[0]
        static_data = []
        
        for cubelet_data in first_snapshot.cubelets_data:
            static_data.append({
                'id': cubelet_data['id']
            })
        
        return static_data
