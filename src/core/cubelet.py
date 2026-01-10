from typing import Literal, List, Dict, Union
from .vector3 import Vector3
from .matrix3x3 import Matrix3x3
import numpy as np


# キューブレットの種類を表す型
CubeletType = Literal["corner", "edge", "center"]

# 座標の許容誤差と許容値
EPSILON_POS = 1e-8
ALLOWED_COORDS = [-1.5, -0.5, 0.5, 1.5]


class Cubelet:
    """個別のキューブレット（立方体の構成要素）を表現するクラス"""

    def __init__(self, id: int, position: Vector3, rotation: Matrix3x3, cubelet_type: CubeletType) -> None:
        """
        キューブレットを初期化します。

        Args:
            id (int): キューブレットの一意識別子
            position (Vector3): キューブレットの位置ベクトル
            rotation (Matrix3x3): キューブレットの回転行列
            cubelet_type (CubeletType): キューブレットの種類（"corner", "edge", "center"）
        """
        self.id = id
        self.position = position
        self.rotation = rotation
        self.type = cubelet_type
        # Store initial position and rotation for tracking
        self.initial_position = Vector3(position.x, position.y, position.z)
        self.initial_rotation = Matrix3x3(rotation.m.copy())

    @staticmethod
    def classify_type(x: float, y: float, z: float) -> CubeletType:
        """
        座標値からキューブレットの種類を判定します。

        Args:
            x (float): x座標
            y (float): y座標
            z (float): z座標

        Returns:
            CubeletType: キューブレットの種類

        Raises:
            ValueError: 座標が有効な組み合わせでない場合
        """
        # 外側の座標値（abs(coord) == 1.5）の数をカウント
        outer_count = sum(1 for coord in [x, y, z] if abs(abs(coord) - 1.5) < EPSILON_POS)

        if outer_count == 3:
            return "corner"
        elif outer_count == 2:
            return "edge"
        elif outer_count == 1:
            return "center"
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}, {z})")

    def apply_rotation(self, rotation_matrix: Matrix3x3) -> None:
        """
        キューブレットに回転を適用します。
        位置ベクトルを変換し、回転行列を更新します（左から乗算）。

        Args:
            rotation_matrix (Matrix3x3): 適用する回転行列
        """
        # 位置ベクトルの回転
        pos_array = self.position.to_numpy()
        new_pos_array = rotation_matrix.apply_to_vector(pos_array)
        self.position = Vector3(new_pos_array[0], new_pos_array[1], new_pos_array[2])

        # 回転行列の更新（左から乗算）
        self.rotation = rotation_matrix @ self.rotation

        # 座標を許容値に丸める
        self.round_position()

    def round_position(self) -> None:
        """
        キューブレットの位置座標を許容値に丸めます。
        各成分について最も近い許容値との差が EPSILON_POS 以下の場合、
        その許容値に置換します。
        
        Note: 浮動小数点演算の累積誤差を防ぐため、回転操作後に呼び出されます。
        """
        pos_array = np.array([self.position.x, self.position.y, self.position.z])
        rounded = np.zeros(3)

        for i, coord in enumerate(pos_array):
            # 最も近い許容値を探す
            closest = min(ALLOWED_COORDS, key=lambda x: abs(x - coord))
            # 差が許容誤差以下なら置換
            if abs(coord - closest) <= EPSILON_POS:
                rounded[i] = closest
            else:
                rounded[i] = coord

        self.position = Vector3(rounded[0], rounded[1], rounded[2])

    def to_dict(self) -> Dict[str, Union[int, str, List[float], List[List[float]]]]:
        """
        キューブレットの状態をJSON出力用の辞書形式に変換します。

        Returns:
            Dict: キューブレットの状態を表す辞書
        """
        return {
            "id": self.id,
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": self.rotation.m.tolist(),
            "type": self.type
        }

    def __repr__(self) -> str:
        """
        キューブレットの文字列表現を返します。

        Returns:
            str: "Cubelet(id=X, type=Y, pos=(x, y, z))" 形式の文字列
        """
        return f"Cubelet(id={self.id}, type={self.type}, pos=({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f}))"