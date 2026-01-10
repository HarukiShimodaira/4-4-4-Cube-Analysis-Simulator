import numpy as np
from numpy import linalg
from typing import Union


class Matrix3x3:
    """3×3回転行列を表現するクラス"""

    # 数値精度定数
    EPSILON_ORTH = 1e-10  # 直交性判定
    EPSILON_DET = 1e-12   # 行列式判定

    def __init__(self, m: np.ndarray) -> None:
        """
        3×3行列を初期化します。

        Args:
            m (np.ndarray): 3×3のNumPy配列

        Raises:
            ValueError: 入力行列のshapeが(3,3)でない場合
        """
        if m.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")
        self.m = m.astype(float)

    @staticmethod
    def identity() -> 'Matrix3x3':
        """
        3×3単位行列を生成します。

        Returns:
            Matrix3x3: 単位行列
        """
        return Matrix3x3(np.eye(3))

    @staticmethod
    def rotation_x(theta: float) -> 'Matrix3x3':
        """
        x軸周りの回転行列を生成します（右手系）。

        Args:
            theta (float): 回転角（度数法）正の値で反時計回り

        Returns:
            Matrix3x3: x軸周りの回転行列
        """
        rad = np.radians(theta)
        cos_t = np.cos(rad)
        sin_t = np.sin(rad)
        return Matrix3x3(np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t]
        ]))

    @staticmethod
    def rotation_y(theta: float) -> 'Matrix3x3':
        """
        y軸周りの回転行列を生成します（右手系）。

        Args:
            theta (float): 回転角（度数法）正の値で反時計回り

        Returns:
            Matrix3x3: y軸周りの回転行列
        """
        rad = np.radians(theta)
        cos_t = np.cos(rad)
        sin_t = np.sin(rad)
        return Matrix3x3(np.array([
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ]))

    @staticmethod
    def rotation_z(theta: float) -> 'Matrix3x3':
        """
        z軸周りの回転行列を生成します（右手系）。

        Args:
            theta (float): 回転角（度数法）正の値で反時計回り

        Returns:
            Matrix3x3: z軸周りの回転行列
        """
        rad = np.radians(theta)
        cos_t = np.cos(rad)
        sin_t = np.sin(rad)
        return Matrix3x3(np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0]
        ]))

    def __matmul__(self, other: 'Matrix3x3') -> 'Matrix3x3':
        """
        行列積を計算します（@演算子）。

        Args:
            other (Matrix3x3): 右側の行列

        Returns:
            Matrix3x3: 行列積の結果
        """
        return Matrix3x3(self.m @ other.m)

    def apply_to_vector(self, v: np.ndarray) -> np.ndarray:
        """
        ベクトルに行列を適用します。

        Args:
            v (np.ndarray): 3次元ベクトル

        Returns:
            np.ndarray: 変換後のベクトル

        Raises:
            ValueError: 入力ベクトルの次元が不正な場合
        """
        if v.shape != (3,):
            raise ValueError("Vector must be 3-dimensional")
        return self.m @ v

    def transpose(self) -> 'Matrix3x3':
        """
        転置行列を返します。

        Returns:
            Matrix3x3: 転置行列
        """
        return Matrix3x3(self.m.T)

    def det(self) -> float:
        """
        行列式を計算します。

        Returns:
            float: 行列式の値
        """
        return float(np.linalg.det(self.m))

    def orthogonalize(self) -> None:
        """
        Gram-Schmidt法（QR分解）で行列を直交化します。
        self.mを直交行列に更新します。
        行列式の符号を保持し、特殊直交群SO(3)を維持します。
        """
        sign = np.sign(np.linalg.det(self.m))
        q, _ = np.linalg.qr(self.m)
        if np.sign(np.linalg.det(q)) != sign:
            q = -q  # 行列式の符号を維持
        self.m = q

    def is_orthogonal(self, epsilon: float = EPSILON_ORTH) -> bool:
        """
        行列が直交行列かどうかを検証します。
        ||R^T R - I||_F < epsilon を確認します。

        Args:
            epsilon (float): 許容誤差

        Returns:
            bool: 直交行列であればTrue
        """
        prod = self.m.T @ self.m
        diff = prod - np.eye(3)
        return np.linalg.norm(diff, ord='fro') < epsilon

    def is_proper_rotation(self, epsilon: float = EPSILON_DET) -> bool:
        """
        行列が特殊直交群SO(3)に属するかを検証します。
        det(R) = 1 を確認します。

        Args:
            epsilon (float): 許容誤差

        Returns:
            bool: 特殊直交行列であればTrue
        """
        return abs(self.det() - 1.0) < epsilon

    def __repr__(self) -> str:
        """
        行列の文字列表現を返します。

        Returns:
            str: 行列の文字列表現
        """
        return f"Matrix3x3(\n{self.m})"