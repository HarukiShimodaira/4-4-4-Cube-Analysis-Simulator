"""
Cubelet Inspector Module
Provides detailed inspection of individual cubelets and edge pairs

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from ..core.cube4x4 import Cube4x4
from ..core.cubelet import Cubelet
from ..core.vector3 import Vector3
from ..core.matrix3x3 import Matrix3x3
from .edge_tracker import EdgeTracker, EdgeInfo


@dataclass
class CubeletState:
    """Individual cubelet state information"""
    id: int
    type: str  # "corner", "edge", "center"
    initial_position: Tuple[float, float, float]
    current_position: Tuple[float, float, float]
    displacement: float  # 初期位置からの移動距離
    rotation_matrix: List[List[float]]  # 3x3回転行列
    rotation_angle: float  # 回転角度（度）
    rotation_axis: Optional[Tuple[float, float, float]]  # 回転軸


@dataclass
class EdgePairInspection:
    """Detailed inspection of an edge pair"""
    pair_id: str
    edge1_id: str
    edge2_id: str
    
    # Edge 1の情報
    edge1_initial_pos: Tuple[float, float, float]
    edge1_current_pos: Tuple[float, float, float]
    edge1_displacement: float
    edge1_rotation_angle: float
    edge1_rotation_axis: Optional[Tuple[float, float, float]]
    
    # Edge 2の情報
    edge2_initial_pos: Tuple[float, float, float]
    edge2_current_pos: Tuple[float, float, float]
    edge2_displacement: float
    edge2_rotation_angle: float
    edge2_rotation_axis: Optional[Tuple[float, float, float]]
    
    # ペア全体の情報
    initial_distance: float
    current_distance: float
    distance_change: float
    same_face: bool
    shared_face: Optional[str]
    
    # 向きの関係
    relative_rotation_angle: float  # 2つのエッジの相対回転角


class CubeletInspector:
    """
    Provides detailed inspection capabilities for individual cubelets and pairs
    """
    
    def __init__(self):
        self.tracker = EdgeTracker()
    
    def get_all_cubelets_state(self, cube: Cube4x4) -> List[CubeletState]:
        """
        全キューブレットの現在状態を取得
        
        Args:
            cube: 調査対象のキューブ
            
        Returns:
            全キューブレットの状態リスト
        """
        states = []
        
        for cubelet in cube.cubelets:
            # 移動距離
            displacement = self._calculate_displacement(
                cubelet.initial_position,
                cubelet.position
            )
            
            # 回転情報（初期回転行列と現在の回転行列から計算）
            angle, axis = self._extract_rotation_info(cubelet.initial_rotation, cubelet.rotation)
            
            state = CubeletState(
                id=cubelet.id,
                type=cubelet.type,
                initial_position=(
                    cubelet.initial_position.x,
                    cubelet.initial_position.y,
                    cubelet.initial_position.z
                ),
                current_position=(
                    cubelet.position.x,
                    cubelet.position.y,
                    cubelet.position.z
                ),
                displacement=displacement,
                rotation_matrix=cubelet.rotation.m.tolist(),
                rotation_angle=angle,
                rotation_axis=axis
            )
            states.append(state)
        
        return states
    
    def inspect_edge_pair(self, cube: Cube4x4, pair_id: str) -> EdgePairInspection:
        """
        特定のエッジペアを詳細に調査
        
        Args:
            cube: 調査対象のキューブ
            pair_id: ペアID（例: "UF", "UR", "DB"など）
            
        Returns:
            エッジペアの詳細情報
            
        Raises:
            ValueError: 無効なペアIDの場合
        """
        # エッジを識別
        edges = self.tracker.identify_edges(cube)
        edge_pairs = self.tracker.get_edge_pairs(edges)
        
        if pair_id not in edge_pairs:
            valid_pairs = list(edge_pairs.keys())
            raise ValueError(f"Invalid pair_id '{pair_id}'. Valid pairs: {valid_pairs}")
        
        edge1, edge2 = edge_pairs[pair_id]
        
        # 各エッジのキューブレット情報を取得
        cubelet1 = self._find_cubelet_by_id(cube, edge1.cubelet_id)
        cubelet2 = self._find_cubelet_by_id(cube, edge2.cubelet_id)
        
        # Edge 1の詳細
        edge1_disp = self._calculate_displacement(edge1.initial_position, edge1.current_position)
        edge1_angle, edge1_axis = self._extract_rotation_info(cubelet1.initial_rotation, cubelet1.rotation)
        
        # Edge 2の詳細
        edge2_disp = self._calculate_displacement(edge2.initial_position, edge2.current_position)
        edge2_angle, edge2_axis = self._extract_rotation_info(cubelet2.initial_rotation, cubelet2.rotation)
        
        # ペア間距離
        initial_dist = self.tracker._distance(edge1.initial_position, edge2.initial_position)
        current_dist = self.tracker._distance(edge1.current_position, edge2.current_position)
        
        # 同一面判定
        shared_face = self.tracker.check_same_face(edge1, edge2)
        
        # 相対回転角（2つのエッジの回転の差）
        relative_angle = abs(edge1_angle - edge2_angle)
        if relative_angle > 180:
            relative_angle = 360 - relative_angle
        
        return EdgePairInspection(
            pair_id=pair_id,
            edge1_id=edge1.edge_id,
            edge2_id=edge2.edge_id,
            
            edge1_initial_pos=(edge1.initial_position.x, edge1.initial_position.y, edge1.initial_position.z),
            edge1_current_pos=(edge1.current_position.x, edge1.current_position.y, edge1.current_position.z),
            edge1_displacement=edge1_disp,
            edge1_rotation_angle=edge1_angle,
            edge1_rotation_axis=edge1_axis,
            
            edge2_initial_pos=(edge2.initial_position.x, edge2.initial_position.y, edge2.initial_position.z),
            edge2_current_pos=(edge2.current_position.x, edge2.current_position.y, edge2.current_position.z),
            edge2_displacement=edge2_disp,
            edge2_rotation_angle=edge2_angle,
            edge2_rotation_axis=edge2_axis,
            
            initial_distance=initial_dist,
            current_distance=current_dist,
            distance_change=current_dist - initial_dist,
            same_face=shared_face is not None,
            shared_face=shared_face,
            
            relative_rotation_angle=relative_angle
        )
    
    def get_edge_pair_list(self) -> List[str]:
        """
        利用可能なエッジペアIDのリストを取得
        
        Returns:
            ペアIDのリスト
        """
        return list(self.tracker.EDGE_PAIRS.keys())
    
    def _find_cubelet_by_id(self, cube: Cube4x4, cubelet_id: int) -> Cubelet:
        """キューブレットIDから該当するキューブレットを検索"""
        for cubelet in cube.cubelets:
            if cubelet.id == cubelet_id:
                return cubelet
        raise ValueError(f"Cubelet with id {cubelet_id} not found")
    
    def _calculate_displacement(self, initial: Vector3, current: Vector3) -> float:
        """初期位置からの移動距離を計算"""
        dx = current.x - initial.x
        dy = current.y - initial.y
        dz = current.z - initial.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)
    
    def _extract_rotation_info(self, initial_rot: Matrix3x3, current_rot: Matrix3x3) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        """
        初期回転行列と現在の回転行列から回転角度と回転軸を計算
        
        Args:
            initial_rot: 初期回転行列
            current_rot: 現在の回転行列
            
        Returns:
            (回転角度[度], 回転軸ベクトル) のタプル
            回転がない場合は (0.0, None)
        """
        # 相対回転行列を計算: R_relative = R_current * R_initial^T
        initial_inv = initial_rot.transpose()
        relative_rot = current_rot @ initial_inv
        
        matrix = relative_rot.m
        
        # トレースから回転角を計算
        trace = np.trace(matrix)
        
        # 単位行列かチェック
        if abs(trace - 3.0) < 1e-6:
            return 0.0, None
        
        # 回転角度
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # 回転軸を計算（Rodriguesの公式の逆）
        if abs(angle_rad) < 1e-6:
            return 0.0, None
        
        # 180度回転の特殊ケース
        if abs(angle_rad - np.pi) < 1e-6:
            # 対角成分から軸を求める
            diag = np.diag(matrix)
            max_idx = np.argmax(diag)
            axis = np.zeros(3)
            axis[max_idx] = np.sqrt((diag[max_idx] + 1) / 2)
            
            # 他の成分を計算
            for i in range(3):
                if i != max_idx:
                    if abs(axis[max_idx]) > 1e-6:
                        axis[i] = matrix[max_idx, i] / (2 * axis[max_idx])
            
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                return angle_deg, tuple(axis)
            return angle_deg, None
        
        # 一般的なケース
        axis = np.array([
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1]
        ])
        
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            return angle_deg, tuple(axis)
        
        return angle_deg, None
    
    def format_inspection_report(self, inspection: EdgePairInspection) -> str:
        """
        検査結果を読みやすいテキストレポートにフォーマット
        
        Args:
            inspection: 検査結果
            
        Returns:
            フォーマットされたレポート文字列
        """
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"エッジペア詳細検査: {inspection.pair_id}")
        lines.append(f"{'='*70}")
        lines.append("")
        
        lines.append(f"【Edge 1: {inspection.edge1_id}】")
        lines.append(f"  初期位置: ({inspection.edge1_initial_pos[0]:6.2f}, {inspection.edge1_initial_pos[1]:6.2f}, {inspection.edge1_initial_pos[2]:6.2f})")
        lines.append(f"  現在位置: ({inspection.edge1_current_pos[0]:6.2f}, {inspection.edge1_current_pos[1]:6.2f}, {inspection.edge1_current_pos[2]:6.2f})")
        lines.append(f"  移動距離: {inspection.edge1_displacement:.4f}")
        lines.append(f"  回転角度: {inspection.edge1_rotation_angle:.2f}°")
        if inspection.edge1_rotation_axis:
            lines.append(f"  回転軸  : ({inspection.edge1_rotation_axis[0]:6.3f}, {inspection.edge1_rotation_axis[1]:6.3f}, {inspection.edge1_rotation_axis[2]:6.3f})")
        lines.append("")
        
        lines.append(f"【Edge 2: {inspection.edge2_id}】")
        lines.append(f"  初期位置: ({inspection.edge2_initial_pos[0]:6.2f}, {inspection.edge2_initial_pos[1]:6.2f}, {inspection.edge2_initial_pos[2]:6.2f})")
        lines.append(f"  現在位置: ({inspection.edge2_current_pos[0]:6.2f}, {inspection.edge2_current_pos[1]:6.2f}, {inspection.edge2_current_pos[2]:6.2f})")
        lines.append(f"  移動距離: {inspection.edge2_displacement:.4f}")
        lines.append(f"  回転角度: {inspection.edge2_rotation_angle:.2f}°")
        if inspection.edge2_rotation_axis:
            lines.append(f"  回転軸  : ({inspection.edge2_rotation_axis[0]:6.3f}, {inspection.edge2_rotation_axis[1]:6.3f}, {inspection.edge2_rotation_axis[2]:6.3f})")
        lines.append("")
        
        lines.append(f"【ペア全体の情報】")
        lines.append(f"  初期距離    : {inspection.initial_distance:.4f}")
        lines.append(f"  現在距離    : {inspection.current_distance:.4f}")
        lines.append(f"  距離変化    : {inspection.distance_change:+.4f}")
        lines.append(f"  同一面      : {'true' if inspection.same_face else 'false'}")
        if inspection.shared_face:
            lines.append(f"  共有面      : {inspection.shared_face}")
        lines.append(f"  相対回転角  : {inspection.relative_rotation_angle:.2f}°")
        lines.append("")
        lines.append(f"{'='*70}")
        
        return "\n".join(lines)
