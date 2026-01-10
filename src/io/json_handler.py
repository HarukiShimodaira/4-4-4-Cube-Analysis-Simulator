import json
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from ..core.cube4x4 import Cube4x4
from ..core.vector3 import Vector3
from ..core.matrix3x3 import Matrix3x3
from ..core.cubelet import Cubelet


def save_cube(cube: Cube4x4, filename: str) -> None:
    """
    キューブの状態をJSONファイルに保存します。

    Args:
        cube (Cube4x4): 保存するキューブ
        filename (str): 保存先のファイル名（.jsonは自動的に付加）

    Raises:
        IOError: ファイルの書き込みに失敗した場合
    """
    # 拡張子の確認と追加
    if not filename.endswith('.json'):
        filename += '.json'

    # キューブの状態を辞書に変換
    cube_state = cube.to_dict()

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cube_state, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save cube state: {str(e)}")


def _create_cubelet_from_dict(data: Dict[str, Any]) -> Cubelet:
    """
    辞書データからCubeletオブジェクトを作成します。

    Args:
        data (Dict[str, Any]): キューブレットのデータ

    Returns:
        Cubelet: 作成されたキューブレット

    Raises:
        ValueError: データ形式が不正な場合
    """
    try:
        # 位置ベクトルの作成
        position = Vector3(*data['position'])

        # 回転行列の作成
        rotation = Matrix3x3(np.array(data['rotation']))

        # キューブレットの作成
        return Cubelet(
            id=data['id'],
            position=position,
            rotation=rotation,
            cubelet_type=data['type']
        )
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid cubelet data format: {str(e)}")


def load_cube(filename: str) -> Cube4x4:
    """
    JSONファイルからキューブの状態を読み込みます。

    Args:
        filename (str): 読み込むJSONファイルのパス（.jsonがない場合は自動的に付加）

    Returns:
        Cube4x4: 読み込まれたキューブ

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: JSONデータの形式が不正な場合
        IOError: ファイルの読み込みに失敗した場合
    """
    # 拡張子の確認と追加
    if not filename.endswith('.json'):
        filename += '.json'

    # ファイルの存在確認
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        # JSONファイルの読み込み
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 基本的なデータ構造の検証
        if not isinstance(data, dict) or 'cubelets' not in data:
            raise ValueError("Invalid JSON format: missing required fields")

        # 新しいキューブを作成
        cube = Cube4x4()
        cube.cubelets.clear()  # 既存のキューブレットをクリア
        cube.rotation_count = data.get('rotation_count', 0)

        # キューブレットの再構築
        for cubelet_data in data['cubelets']:
            cubelet = _create_cubelet_from_dict(cubelet_data)
            cube.cubelets.append(cubelet)

        # キューブの状態を検証
        valid, errors = cube.validate()
        if not valid:
            raise ValueError(f"Invalid cube state after loading: {errors}")

        return cube

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to load cube state: {str(e)}")


# Aliases for backward compatibility
save_cube_state = save_cube
load_cube_state = load_cube