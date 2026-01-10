#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4×4×4 Cube Analysis Simulator - Core Module

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

from typing import List, Tuple, Literal, Dict, Any
from .vector3 import Vector3
from .matrix3x3 import Matrix3x3
from .cubelet import Cubelet
import numpy as np
import itertools


# 座標値と許容誤差の定数
COORDS = [-1.5, -0.5, 0.5, 1.5]
EPSILON_POS = 1e-8


class Cube4x4:
    """4×4×4ルービックキューブを表現するクラス"""

    def __init__(self) -> None:
        """
        4×4×4ルービックキューブを初期化します。
        56個のキューブレットを生成し、回転カウントを0に設定します。
        """
        self.cubelets: List[Cubelet] = []
        self.rotation_count: int = 0
        self._initialize()

    def _initialize(self) -> None:
        """
        キューブの初期状態を生成します。
        - 56個のキューブレットを正しい位置に配置
        - 内部の8個のキューブレットは除外
        - 各キューブレットのタイプを判定
        """
        cubelet_id = 0
        # 3次元グリッドを生成
        for x, y, z in itertools.product(COORDS, COORDS, COORDS):
            # 内部キューブレット（全座標のabsが0.5）を除外
            if abs(abs(x) - 0.5) < EPSILON_POS and \
               abs(abs(y) - 0.5) < EPSILON_POS and \
               abs(abs(z) - 0.5) < EPSILON_POS:
                continue

            position = Vector3(x, y, z)
            rotation = Matrix3x3.identity()
            cubelet_type = Cubelet.classify_type(x, y, z)
            
            self.cubelets.append(Cubelet(cubelet_id, position, rotation, cubelet_type))
            cubelet_id += 1

    def _select_layer(self, axis: Literal['x', 'y', 'z'], value: float) -> List[Cubelet]:
        """
        指定された層に属するキューブレットを抽出します。

        Args:
            axis (Literal['x','y','z']): 軸の指定
            value (float): 層の座標値

        Returns:
            List[Cubelet]: 指定層のキューブレット
        """
        def check_coord(cubelet: Cubelet) -> bool:
            pos = cubelet.position
            coord = getattr(pos, axis)
            return abs(coord - value) < EPSILON_POS

        return list(filter(check_coord, self.cubelets))

    def _rotate_layer(self, axis: Literal['x', 'y', 'z'], value: float, angle: float) -> None:
        """
        指定された層を回転させます。

        Args:
            axis (Literal['x','y','z']): 回転軸
            value (float): 層の座標値
            angle (float): 回転角度（度数法）
        """
        # 対象キューブレットの選択
        target_cubelets = self._select_layer(axis, value)

        # 回転行列の生成
        rotation_func = getattr(Matrix3x3, f"rotation_{axis}")
        rotation_matrix = rotation_func(angle)

        # 対象キューブレットの回転
        for cubelet in target_cubelets:
            cubelet.apply_rotation(rotation_matrix)

        # 回転カウントの更新
        self.rotation_count += 1

    # 面の回転メソッド群
    def R(self) -> None:
        """R面の90度時計回り回転（外側から見て）"""
        self._rotate_layer('x', 1.5, -90)

    def Rp(self) -> None:
        """R面の90度反時計回り回転（外側から見て）"""
        self._rotate_layer('x', 1.5, 90)

    def R2(self) -> None:
        """R面の180度回転"""
        self._rotate_layer('x', 1.5, 180)

    def L(self) -> None:
        """L面の90度時計回り回転（外側から見て）"""
        self._rotate_layer('x', -1.5, 90)

    def Lp(self) -> None:
        """L面の90度反時計回り回転（外側から見て）"""
        self._rotate_layer('x', -1.5, -90)

    def L2(self) -> None:
        """L面の180度回転"""
        self._rotate_layer('x', -1.5, 180)

    def U(self) -> None:
        """U面の90度時計回り回転（上から見て）"""
        self._rotate_layer('y', 1.5, -90)

    def Up(self) -> None:
        """U面の90度反時計回り回転（上から見て）"""
        self._rotate_layer('y', 1.5, 90)

    def U2(self) -> None:
        """U面の180度回転"""
        self._rotate_layer('y', 1.5, 180)

    def D(self) -> None:
        """D面の90度時計回り回転（下から見て）"""
        self._rotate_layer('y', -1.5, 90)

    def Dp(self) -> None:
        """D面の90度反時計回り回転（下から見て）"""
        self._rotate_layer('y', -1.5, -90)

    def D2(self) -> None:
        """D面の180度回転"""
        self._rotate_layer('y', -1.5, 180)

    def F(self) -> None:
        """F面の90度時計回り回転（前から見て）"""
        self._rotate_layer('z', 1.5, -90)

    def Fp(self) -> None:
        """F面の90度反時計回り回転（前から見て）"""
        self._rotate_layer('z', 1.5, 90)

    def F2(self) -> None:
        """F面の180度回転"""
        self._rotate_layer('z', 1.5, 180)

    def B(self) -> None:
        """B面の90度時計回り回転（後ろから見て）"""
        self._rotate_layer('z', -1.5, 90)

    def Bp(self) -> None:
        """B面の90度反時計回り回転（後ろから見て）"""
        self._rotate_layer('z', -1.5, -90)

    def B2(self) -> None:
        """B面の180度回転"""
        self._rotate_layer('z', -1.5, 180)

    # 内側層の回転（小文字）
    def r(self) -> None:
        """右側内側層の90度時計回り回転（外側から見て）"""
        self._rotate_layer('x', 0.5, -90)

    def rp(self) -> None:
        """右側内側層の90度反時計回り回転（外側から見て）"""
        self._rotate_layer('x', 0.5, 90)

    def r2(self) -> None:
        """右側内側層の180度回転"""
        self._rotate_layer('x', 0.5, 180)

    def l(self) -> None:
        """左側内側層の90度時計回り回転（外側から見て）"""
        self._rotate_layer('x', -0.5, 90)

    def lp(self) -> None:
        """左側内側層の90度反時計回り回転（外側から見て）"""
        self._rotate_layer('x', -0.5, -90)

    def l2(self) -> None:
        """左側内側層の180度回転"""
        self._rotate_layer('x', -0.5, 180)

    def u(self) -> None:
        """上側内側層の90度時計回り回転（上から見て）"""
        self._rotate_layer('y', 0.5, -90)

    def up(self) -> None:
        """上側内側層の90度反時計回り回転（上から見て）"""
        self._rotate_layer('y', 0.5, 90)

    def u2(self) -> None:
        """上側内側層の180度回転"""
        self._rotate_layer('y', 0.5, 180)

    def d(self) -> None:
        """下側内側層の90度時計回り回転（下から見て）"""
        self._rotate_layer('y', -0.5, 90)

    def dp(self) -> None:
        """下側内側層の90度反時計回り回転（下から見て）"""
        self._rotate_layer('y', -0.5, -90)

    def d2(self) -> None:
        """下側内側層の180度回転"""
        self._rotate_layer('y', -0.5, 180)

    def f(self) -> None:
        """前側内側層の90度時計回り回転（前から見て）"""
        self._rotate_layer('z', 0.5, -90)

    def fp(self) -> None:
        """前側内側層の90度反時計回り回転（前から見て）"""
        self._rotate_layer('z', 0.5, 90)

    def f2(self) -> None:
        """前側内側層の180度回転"""
        self._rotate_layer('z', 0.5, 180)

    def b(self) -> None:
        """後側内側層の90度時計回り回転（後ろから見て）"""
        self._rotate_layer('z', -0.5, 90)

    def bp(self) -> None:
        """後側内側層の90度反時計回り回転（後ろから見て）"""
        self._rotate_layer('z', -0.5, -90)

    def b2(self) -> None:
        """後側内側層の180度回転"""
        self._rotate_layer('z', -0.5, 180)

    # 組み合わせ操作（ワイド回転）
    def Rw(self) -> None:
        """R面とr層のワイド回転（2層同時に時計回り）"""
        self.R()
        self.r()

    def Rwp(self) -> None:
        """R面とr層のワイド回転（2層同時に反時計回り）"""
        self.Rp()
        self.rp()

    def Rw2(self) -> None:
        """R面とr層のワイド回転（2層同時に180度）"""
        self.R2()
        self.r2()

    def Lw(self) -> None:
        """L面とl層のワイド回転（2層同時に時計回り）"""
        self.L()
        self.l()

    def Lwp(self) -> None:
        """L面とl層のワイド回転（2層同時に反時計回り）"""
        self.Lp()
        self.lp()

    def Lw2(self) -> None:
        """L面とl層のワイド回転（2層同時に180度）"""
        self.L2()
        self.l2()

    def Uw(self) -> None:
        """U面とu層のワイド回転（2層同時に時計回り）"""
        self.U()
        self.u()

    def Uwp(self) -> None:
        """U面とu層のワイド回転（2層同時に反時計回り）"""
        self.Up()
        self.up()

    def Uw2(self) -> None:
        """U面とu層のワイド回転（2層同時に180度）"""
        self.U2()
        self.u2()

    def Dw(self) -> None:
        """D面とd層のワイド回転（2層同時に時計回り）"""
        self.D()
        self.d()

    def Dwp(self) -> None:
        """D面とd層のワイド回転（2層同時に反時計回り）"""
        self.Dp()
        self.dp()

    def Dw2(self) -> None:
        """D面とd層のワイド回転（2層同時に180度）"""
        self.D2()
        self.d2()

    def Fw(self) -> None:
        """F面とf層のワイド回転（2層同時に時計回り）"""
        self.F()
        self.f()

    def Fwp(self) -> None:
        """F面とf層のワイド回転（2層同時に反時計回り）"""
        self.Fp()
        self.fp()

    def Fw2(self) -> None:
        """F面とf層のワイド回転（2層同時に180度）"""
        self.F2()
        self.f2()

    def Bw(self) -> None:
        """B面とb層のワイド回転（2層同時に時計回り）"""
        self.B()
        self.b()

    def Bwp(self) -> None:
        """B面とb層のワイド回転（2層同時に反時計回り）"""
        self.Bp()
        self.bp()

    def Bw2(self) -> None:
        """B面とb層のワイド回転（2層同時に180度）"""
        self.B2()
        self.b2()

    # スライス回転（M, E, S）
    def M(self) -> None:
        """Middle層の回転（L方向、x = -0.5 と 0.5）"""
        self.lp()
        self.l()

    def Mp(self) -> None:
        """Middle層の逆回転"""
        self.l()
        self.lp()

    def M2(self) -> None:
        """Middle層の180度回転"""
        self.l2()
        self.l2()

    def E(self) -> None:
        """Equator層の回転（D方向、y = -0.5 と 0.5）"""
        self.dp()
        self.d()

    def Ep(self) -> None:
        """Equator層の逆回転"""
        self.d()
        self.dp()

    def E2(self) -> None:
        """Equator層の180度回転"""
        self.d2()
        self.d2()

    def S(self) -> None:
        """Standing層の回転（F方向、z = -0.5 と 0.5）"""
        self.fp()
        self.f()

    def Sp(self) -> None:
        """Standing層の逆回転"""
        self.f()
        self.fp()

    def S2(self) -> None:
        """Standing層の180度回転"""
        self.f2()
        self.f2()

    # キューブ全体の回転
    def x(self) -> None:
        """キューブ全体をx軸周りに回転（R方向）"""
        self.R()
        self.Mp()
        self.Lp()

    def xp(self) -> None:
        """キューブ全体をx軸周りに逆回転"""
        self.Rp()
        self.M()
        self.L()

    def x2(self) -> None:
        """キューブ全体をx軸周りに180度回転"""
        self.R2()
        self.M2()
        self.L2()

    def y(self) -> None:
        """キューブ全体をy軸周りに回転（U方向）"""
        self.U()
        self.Ep()
        self.Dp()

    def yp(self) -> None:
        """キューブ全体をy軸周りに逆回転"""
        self.Up()
        self.E()
        self.D()

    def y2(self) -> None:
        """キューブ全体をy軸周りに180度回転"""
        self.U2()
        self.E2()
        self.D2()

    def z(self) -> None:
        """キューブ全体をz軸周りに回転（F方向）"""
        self.F()
        self.Sp()
        self.Bp()

    def zp(self) -> None:
        """キューブ全体をz軸周りに逆回転"""
        self.Fp()
        self.S()
        self.B()

    def z2(self) -> None:
        """キューブ全体をz軸周りに180度回転"""
        self.F2()
        self.S2()
        self.B2()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        キューブの状態を検証します。

        以下の条件をチェックします：
        1. 位置の一意性（キューブレット間の距離 > 0.9）
        2. 回転行列の行列式 = 1
        3. タイプ別の個数が正しいか

        Returns:
            Tuple[bool, List[str]]: (検証結果, エラーメッセージのリスト)
        """
        errors = []

        # 1. 位置の一意性チェック
        for i, c1 in enumerate(self.cubelets):
            for c2 in self.cubelets[i + 1:]:
                dx = c1.position.x - c2.position.x
                dy = c1.position.y - c2.position.y
                dz = c1.position.z - c2.position.z
                distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                if distance < 0.9:
                    errors.append(f"Cubelets {c1.id} and {c2.id} are too close: distance = {distance:.6f}")

        # 2. 回転行列の検証
        for cubelet in self.cubelets:
            if not cubelet.rotation.is_proper_rotation():
                errors.append(f"Cubelet {cubelet.id} has invalid rotation matrix")

        # 3. タイプ別個数の検証
        type_counts = {
            "corner": sum(1 for c in self.cubelets if c.type == "corner"),
            "edge": sum(1 for c in self.cubelets if c.type == "edge"),
            "center": sum(1 for c in self.cubelets if c.type == "center")
        }

        expected_counts = {"corner": 8, "edge": 24, "center": 24}
        for ctype, count in type_counts.items():
            if count != expected_counts[ctype]:
                errors.append(f"Invalid {ctype} count: {count} (expected {expected_counts[ctype]})")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """
        キューブの状態を辞書形式で返します。

        Returns:
            Dict[str, Any]: キューブの状態を表す辞書
        """
        return {
            "rotation_count": self.rotation_count,
            "cubelets": [cubelet.to_dict() for cubelet in self.cubelets]
        }

    def __repr__(self) -> str:
        """
        キューブの文字列表現を返します。

        Returns:
            str: キューブの状態を表す文字列
        """
        valid, errors = self.validate()
        status = "valid" if valid else "invalid"
        return f"Cube4x4(rotation_count={self.rotation_count}, status={status}, cubelets={len(self.cubelets)})"