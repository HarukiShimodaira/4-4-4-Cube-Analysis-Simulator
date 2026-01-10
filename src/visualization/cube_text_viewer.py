from typing import List, Dict
import numpy as np
from ..core.cube4x4 import Cube4x4
from ..core.vector3 import Vector3

# 色と文字のマッピング、およびANSIエスケープコード
COLOR_CODES = {
    'W': '\033[97;1m',        # 明るい白（太字）
    'Y': '\033[93;1m',        # 明るい黄色（太字）
    'G': '\033[92;1m',        # 明るい緑（太字）
    'B': '\033[94;1m',        # 明るい青（太字）
    'R': '\033[91;1m',        # 明るい赤（太字）
    'O': '\033[38;5;214;1m',  # 明るいオレンジ（太字）
}

RESET_COLOR = '\033[0m'  # リセットコード
BORDER_COLOR = '\033[90m'  # グレー（境界線用）


def get_face_grid(cube: Cube4x4, face: str) -> List[List[str]]:
    """
    指定された面のグリッドを取得します。

    Args:
        cube (Cube4x4): キューブオブジェクト
        face (str): 面の指定（'U', 'D', 'F', 'B', 'R', 'L'）

    Returns:
        List[List[str]]: 4x4のグリッド（文字の2次元配列）
    """
    # 面の座標を定義
    coords = {
        'U': (lambda x, z: (x, 1.5, z)),      # 上面：y = 1.5
        'D': (lambda x, z: (x, -1.5, z)),     # 下面：y = -1.5
        'F': (lambda x, y: (x, y, 1.5)),      # 前面：z = 1.5
        'B': (lambda x, y: (x, y, -1.5)),     # 後面：z = -1.5
        'R': (lambda y, z: (1.5, y, z)),      # 右面：x = 1.5
        'L': (lambda y, z: (-1.5, y, z))      # 左面：x = -1.5
    }

    # 面の基準方向を定義
    face_directions = {
        'U': np.array([0, 1, 0]),
        'D': np.array([0, -1, 0]),
        'F': np.array([0, 0, 1]),
        'B': np.array([0, 0, -1]),
        'R': np.array([1, 0, 0]),
        'L': np.array([-1, 0, 0])
    }

    # グリッドを初期化
    grid = [['.' for _ in range(4)] for _ in range(4)]
    
    # 座標変換関数を取得
    coord_func = coords[face]
    face_normal = face_directions[face]
    
    # 座標値の配列
    values = [-1.5, -0.5, 0.5, 1.5]
    
    # グリッドを埋める
    # valuesインデックス: 0=-1.5, 1=-0.5, 2=0.5, 3=1.5
    for i in range(4):
        for j in range(4):
            if face == 'U':
                # U面(y=1.5): 上から見て i=行(z軸), j=列(x軸)
                # i=0→z=-1.5(奥), i=3→z=1.5(手前), j=0→x=-1.5(左), j=3→x=1.5(右)
                x = values[j]
                z = values[i]
                y = 1.5
            elif face == 'D':
                # D面(y=-1.5): 下から見て i=行(z軸), j=列(x軸)
                # i=0→z=1.5(手前), i=3→z=-1.5(奥), j=0→x=-1.5(左), j=3→x=1.5(右)
                x = values[j]
                z = values[3 - i]  # 逆順
                y = -1.5
            elif face == 'F':
                # F面(z=1.5): 前から見て i=行(y軸), j=列(x軸)
                # i=0→y=1.5(上), i=3→y=-1.5(下), j=0→x=-1.5(左), j=3→x=1.5(右)
                x = values[j]
                y = values[3 - i]  # 逆順
                z = 1.5
            elif face == 'B':
                # B面(z=-1.5): 後ろから見て i=行(y軸), j=列(x軸)
                # i=0→y=1.5(上), i=3→y=-1.5(下), j=0→x=1.5(右), j=3→x=-1.5(左)
                x = values[3 - j]  # 逆順
                y = values[3 - i]  # 逆順
                z = -1.5
            elif face == 'R':
                # R面(x=1.5): 右から見て i=行(y軸), j=列(z軸)
                # i=0→y=1.5(上), i=3→y=-1.5(下), j=0→z=1.5(手前), j=3→z=-1.5(奥)
                z = values[3 - j]  # 逆順
                y = values[3 - i]  # 逆順
                x = 1.5
            else:  # L
                # L面(x=-1.5): 左から見て i=行(y軸), j=列(z軸)
                # i=0→y=1.5(上), i=3→y=-1.5(下), j=0→z=-1.5(奥), j=3→z=1.5(手前)
                z = values[j]
                y = values[3 - i]  # 逆順
                x = -1.5
            
            # この位置のキューブレットを探す
            target_pos = Vector3(x, y, z)
            for cubelet in cube.cubelets:
                if (abs(cubelet.position.x - x) < 1e-6 and
                    abs(cubelet.position.y - y) < 1e-6 and
                    abs(cubelet.position.z - z) < 1e-6):
                    # 現在見ている面の方向を逆回転して、初期状態での方向を確認
                    # 回転行列の逆行列（転置）を適用
                    rotation_inv = cubelet.rotation.m.T
                    original_direction = rotation_inv @ face_normal
                    
                    # 最も近い軸方向を見つける
                    max_dot = -1
                    best_match = None
                    for direction, normal in face_directions.items():
                        dot = np.dot(original_direction, normal)
                        if dot > max_dot:
                            max_dot = dot
                            best_match = direction
                    
                    # 色を文字に変換
                    if best_match == 'U':
                        grid[i][j] = 'W'
                    elif best_match == 'D':
                        grid[i][j] = 'Y'
                    elif best_match == 'F':
                        grid[i][j] = 'G'
                    elif best_match == 'B':
                        grid[i][j] = 'B'
                    elif best_match == 'R':
                        grid[i][j] = 'R'
                    elif best_match == 'L':
                        grid[i][j] = 'O'
                    break
    
    return grid


def print_cube_state(cube: Cube4x4) -> None:
    """
    キューブの状態をテキストで表示します。

    Args:
        cube (Cube4x4): 表示するキューブ
    
    展開図の形式:
           U
        L  F  R  B
           D
    """
    # 各面のグリッドを取得
    faces = {
        'U': get_face_grid(cube, 'U'),
        'D': get_face_grid(cube, 'D'),
        'F': get_face_grid(cube, 'F'),
        'R': get_face_grid(cube, 'R'),
        'B': get_face_grid(cube, 'B'),
        'L': get_face_grid(cube, 'L')
    }
    
    # 画面をクリア
    print('\033[2J\033[H', end='')
    
    # 上面を表示
    print("       U")
    for row in faces['U']:
        colored_row = [f"{COLOR_CODES[c]}■{RESET_COLOR}" for c in row]
        print("     " + " ".join(colored_row))
    print()
    
    # 中央の4面を表示
    print("   L      F      R      B")
    for i in range(4):
        row = []
        for face in ['L', 'F', 'R', 'B']:
            colored_chars = [f"{COLOR_CODES[c]}■{RESET_COLOR}" for c in faces[face][i]]
            row.extend(colored_chars)
            row.append(' ')
        print("  " + " ".join(row))
    print()
    
    # 下面を表示
    print("       D")
    for row in faces['D']:
        colored_row = [f"{COLOR_CODES[c]}■{RESET_COLOR}" for c in row]
        print("     " + " ".join(colored_row))


if __name__ == '__main__':
    # テスト用
    cube = Cube4x4()
    print("Initial State")
    print_cube_state(cube)
    
    print("\nAfter R Operation")
    cube.R()
    print_cube_state(cube)