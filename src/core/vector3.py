import numpy as np
from typing import Union


class Vector3:
    """3次元ベクトルを表現するクラス"""

    def __init__(self, x: float, y: float, z: float) -> None:
        """
        3次元ベクトルを初期化します。

        Args:
            x (float): x成分
            y (float): y成分
            z (float): z成分
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def to_numpy(self) -> np.ndarray:
        """
        ベクトルをNumPy配列として返します。

        Returns:
            np.ndarray: [x, y, z]の形式のNumPy配列
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    def __add__(self, other: 'Vector3') -> 'Vector3':
        """
        ベクトルの加算を行います。

        Args:
            other (Vector3): 加算するベクトル

        Returns:
            Vector3: 加算結果のベクトル
        """
        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """
        ベクトルの減算を行います。

        Args:
            other (Vector3): 減算するベクトル

        Returns:
            Vector3: 減算結果のベクトル
        """
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def dot(self, other: 'Vector3') -> float:
        """
        ベクトルの内積を計算します。

        Args:
            other (Vector3): 内積を計算する対象のベクトル

        Returns:
            float: 内積の結果
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        """
        ベクトルのノルム（長さ）を計算します。

        Returns:
            float: ベクトルのノルム
        """
        return np.sqrt(self.dot(self))

    def magnitude(self) -> float:
        """
        ベクトルの大きさを計算します（normのエイリアス）。

        Returns:
            float: ベクトルの大きさ
        """
        return self.norm()

    def normalize(self) -> 'Vector3':
        """
        ベクトルを正規化（単位ベクトル化）します。

        Returns:
            Vector3: 正規化されたベクトル
        """
        norm = self.norm()
        if norm == 0:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(
            self.x / norm,
            self.y / norm,
            self.z / norm
        )

    def __repr__(self) -> str:
        """
        ベクトルの文字列表現を返します。

        Returns:
            str: "Vector3(x, y, z)"形式の文字列
        """
        return f"Vector3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"
