from typing import Dict, List, Tuple
from ..core.cube4x4 import Cube4x4
import numpy as np


def _get_edge_positions(cube: Cube4x4) -> List[Tuple[float, float, float]]:
    """エッジキューブレットの位置を取得します"""
    return [(c.position.x, c.position.y, c.position.z)
            for c in cube.cubelets if c.type == "edge"]


def _get_corner_positions(cube: Cube4x4) -> List[Tuple[float, float, float]]:
    """コーナーキューブレットの位置を取得します"""
    return [(c.position.x, c.position.y, c.position.z)
            for c in cube.cubelets if c.type == "corner"]


def _get_center_orientations(cube: Cube4x4) -> List[np.ndarray]:
    """センターキューブレットの向きを取得します"""
    return [c.rotation.m for c in cube.cubelets if c.type == "center"]


def _count_inversions(positions: List[Tuple[float, float, float]]) -> int:
    """
    位置の順列における転倒数を計算します。
    転倒数が偶数ならパリティなし、奇数ならパリティあり。

    Args:
        positions: 座標のリスト [(x, y, z), ...]

    Returns:
        int: 転倒数
    """
    # 3次元座標を1次元インデックスに変換
    def pos_to_index(pos: Tuple[float, float, float]) -> int:
        x, y, z = pos
        # 座標を正の整数にマッピング
        ix = int((x + 1.5) * 2)
        iy = int((y + 1.5) * 2)
        iz = int((z + 1.5) * 2)
        return ix * 16 + iy * 4 + iz

    # 位置をインデックスに変換
    indices = [pos_to_index(pos) for pos in positions]
    
    # 転倒数を計算
    inversions = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                inversions += 1
    
    return inversions


def detect_edge_parity(cube: Cube4x4) -> bool:
    """
    エッジの位置パリティを検出します。
    パリティがある場合はTrue、ない場合はFalseを返します。
    
    Args:
        cube (Cube4x4): パリティをチェックするキューブ

    Returns:
        bool: パリティの有無
    """
    edge_positions = _get_edge_positions(cube)
    inversions = _count_inversions(edge_positions)
    return (inversions % 2) == 1


def detect_corner_parity(cube: Cube4x4) -> bool:
    """
    コーナーの位置パリティを検出します。
    パリティがある場合はTrue、ない場合はFalseを返します。
    
    Args:
        cube (Cube4x4): パリティをチェックするキューブ

    Returns:
        bool: パリティの有無
    """
    corner_positions = _get_corner_positions(cube)
    inversions = _count_inversions(corner_positions)
    return (inversions % 2) == 1


def detect_center_parity(cube: Cube4x4) -> bool:
    """
    センターの向きパリティを検出します。
    回転が90度または270度のセンターが奇数個ある場合はTrue、
    それ以外の場合はFalseを返します。
    
    Args:
        cube (Cube4x4): パリティをチェックするキューブ

    Returns:
        bool: パリティの有無
    """
    center_orientations = _get_center_orientations(cube)
    odd_rotations = 0

    # 各センターの回転角を判定
    for rot_matrix in center_orientations:
        # 回転角を求める（arccos(tr(R)-1)/2）
        trace = np.trace(rot_matrix)
        if abs(trace - 3.0) > 1e-6:  # 単位行列でない場合
            angle = np.arccos((trace - 1) / 2)
            angle_deg = np.degrees(angle)
            
            # 90度または270度の回転を検出
            if abs(angle_deg - 90) < 1e-6 or abs(angle_deg - 270) < 1e-6:
                odd_rotations += 1

    return (odd_rotations % 2) == 1


def full_parity_check(cube: Cube4x4) -> Dict[str, bool]:
    """
    キューブの全パリティ状態をチェックします。
    
    Args:
        cube (Cube4x4): パリティをチェックするキューブ

    Returns:
        Dict[str, bool]: 各パリティの状態を示す辞書
            {
                "edge_parity": bool,   # エッジの位置パリティ
                "corner_parity": bool, # コーナーの位置パリティ
                "center_parity": bool  # センターの向きパリティ
            }
    """
    return {
        "edge_parity": detect_edge_parity(cube),
        "corner_parity": detect_corner_parity(cube),
        "center_parity": detect_center_parity(cube)
    }