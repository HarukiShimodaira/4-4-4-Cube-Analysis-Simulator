import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict
from ..core.cube4x4 import Cube4x4
from ..core.vector3 import Vector3
from ..core.matrix3x3 import Matrix3x3


# キューブの色定義
COLORS = {
    'U': '#FFFFFF',  # 上面（白）
    'D': '#FFFF00',  # 下面（黄）
    'F': '#00FF00',  # 前面（緑）
    'B': '#0000FF',  # 背面（青）
    'R': '#FF0000',  # 右面（赤）
    'L': '#FFA500'   # 左面（オレンジ）
}


def _create_unit_cube(center: Vector3, size: float = 0.8) -> Tuple[np.ndarray, List[List[int]], Dict[str, List[int]]]:
    """
    単位キューブの頂点と面を生成します。

    Args:
        center (Vector3): キューブの中心座標
        size (float): キューブの大きさ（デフォルト0.8でギャップを作る）

    Returns:
        Tuple[np.ndarray, List[List[int]], Dict[str, List[int]]]:
            - 頂点座標の配列
            - 面を構成する頂点インデックスのリスト
            - 方向ごとの面のインデックス
    """
    # 頂点の相対座標
    half = size / 2
    vertices = np.array([
        [-half, -half, -half],  # 0: 左下後
        [half, -half, -half],   # 1: 右下後
        [half, half, -half],    # 2: 右上後
        [-half, half, -half],   # 3: 左上後
        [-half, -half, half],   # 4: 左下前
        [half, -half, half],    # 5: 右下前
        [half, half, half],     # 6: 右上前
        [-half, half, half]     # 7: 左上前
    ])

    # 中心位置を加算
    vertices += np.array([center.x, center.y, center.z])

    # 面の定義（頂点インデックス）
    faces = [
        [0, 1, 2, 3],  # 後面（B）
        [4, 5, 6, 7],  # 前面（F）
        [0, 4, 7, 3],  # 左面（L）
        [1, 5, 6, 2],  # 右面（R）
        [3, 2, 6, 7],  # 上面（U）
        [0, 1, 5, 4]   # 下面（D）
    ]

    # 方向ごとの面のマッピング
    face_mapping = {
        'B': 0,
        'F': 1,
        'L': 2,
        'R': 3,
        'U': 4,
        'D': 5
    }

    return vertices, faces, face_mapping


def _get_face_colors(position: Vector3, rotation: Matrix3x3) -> Dict[str, str]:
    """
    キューブレットの位置と回転から、外側から見える面の色を決定します。

    Args:
        position (Vector3): キューブレットの位置
        rotation (Matrix3x3): キューブレットの回転行列

    Returns:
        Dict[str, str]: 方向ごとの色のマッピング
    """
    # 基本方向ベクトル（NumPy配列として定義）
    directions = {
        'R': np.array([1, 0, 0]),
        'L': np.array([-1, 0, 0]),
        'U': np.array([0, 1, 0]),
        'D': np.array([0, -1, 0]),
        'F': np.array([0, 0, 1]),
        'B': np.array([0, 0, -1])
    }

    # 位置から外側の面を判定（絶対値が1.5の座標の方向が外側）
    outer_faces = {
        'R': abs(position.x - 1.5) < 1e-6,
        'L': abs(position.x + 1.5) < 1e-6,
        'U': abs(position.y - 1.5) < 1e-6,
        'D': abs(position.y + 1.5) < 1e-6,
        'F': abs(position.z - 1.5) < 1e-6,
        'B': abs(position.z + 1.5) < 1e-6
    }

    # 色の割り当て
    colors = {}
    for face, direction in directions.items():
        if outer_faces[face]:
            # 外側の面の場合は色を割り当て
            # 回転行列の逆行列（転置）を使って、初期状態での方向を確認
            rotation_inv = rotation.m.T
            original_direction = rotation_inv @ direction
            
            # 最も近い基本方向を見つける
            max_dot = -1
            best_match = None
            for base_face, base_dir in directions.items():
                dot = np.dot(original_direction, base_dir)
                if dot > max_dot:
                    max_dot = dot
                    best_match = base_face
            colors[face] = COLORS[best_match]
        else:
            # 内側の面は灰色（または非表示）
            colors[face] = '#DDDDDD'

    return colors


_current_fig = None
_current_ax = None
_current_view = {'elev': 30, 'azim': -45}  # 現在の視点を辞書で保存


def visualize_cube(cube: Cube4x4, update: bool = False, show: bool = True, elev: float = None, azim: float = None):
    """
    4×4×4ルービックキューブを3Dで表示します。

    Args:
        cube (Cube4x4): 表示するキューブ
        update (bool): Trueの場合、既存の図を更新します
        show (bool): Trueの場合、図を表示します。Falseの場合は図オブジェクトを返します。
        elev (float): 仰角（デフォルト30）
        azim (float): 方位角（デフォルト-45）
    
    Returns:
        図オブジェクト（show=Falseの場合）またはNone
    """
    global _current_fig, _current_ax, _current_view
    
    # 視点が指定されている場合は更新
    if elev is not None:
        _current_view['elev'] = elev
    if azim is not None:
        _current_view['azim'] = azim

    if update and _current_fig is not None and _current_ax is not None:
        # 現在の視点を保存（clear前に取得）
        # matplotlibのインタラクティブな回転後の実際の視点を取得
        try:
            _current_view['elev'] = _current_ax.elev
            _current_view['azim'] = _current_ax.azim
        except Exception:
            pass  # 取得できない場合は現在の値を維持
        # 既存の図をクリア
        _current_ax.clear()
    elif not update or _current_fig is None:
        # 新しい3D図を作成
        _current_fig = plt.figure(figsize=(10, 10))
        _current_ax = _current_fig.add_subplot(111, projection='3d')
        plt.ion()  # インタラクティブモードを有効化
    
    # キューブを描画してから視点を設定
    # （先に描画しないとview_initが正しく動作しない場合がある）

    # 各キューブレットを描画
    for cubelet in cube.cubelets:
        # キューブレットの頂点と面を生成
        vertices, faces, face_mapping = _create_unit_cube(cubelet.position)
        
        # 面の色を取得
        colors = _get_face_colors(cubelet.position, cubelet.rotation)

        # 各面を描画
        for direction, face_idx in face_mapping.items():
            face_vertices = vertices[faces[face_idx]]
            
            # 面を作成して色を設定
            poly = Poly3DCollection([face_vertices], alpha=1.0)
            poly.set_facecolor(colors[direction])
            poly.set_edgecolor('black')
            _current_ax.add_collection3d(poly)

    # 軸の設定
    _current_ax.set_xlabel('X')
    _current_ax.set_ylabel('Y')
    _current_ax.set_zlabel('Z')
    
    # 表示範囲の設定
    _current_ax.set_xlim(-2, 2)
    _current_ax.set_ylim(-2, 2)
    _current_ax.set_zlim(-2, 2)
    
    # アスペクト比を1:1:1に設定
    _current_ax.set_box_aspect([1, 1, 1])
    
    # グリッドの表示
    _current_ax.grid(True)
    
    # タイトルの設定
    plt.title('4×4×4 Cube Analysis Simulator')
    
    # 視点の設定（描画後に設定）
    _current_ax.view_init(elev=_current_view['elev'], azim=_current_view['azim'])
    
    if not show:
        # 図オブジェクトを返す（表示しない）
        return _current_fig
    elif update:
        # 図の更新
        _current_fig.canvas.draw()
        _current_fig.canvas.flush_events()
    else:
        # 初回表示
        plt.show()
    
    return None


def save_cube_image(cube: Cube4x4, filename: str) -> None:
    """
    キューブの3D表示を画像ファイルとして保存します。

    Args:
        cube (Cube4x4): 保存するキューブ
        filename (str): 保存先のファイル名（.png等の拡張子を含む）
    """
    # 新しい3D図を作成
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 視点の設定
    ax.view_init(elev=20, azim=45)

    # 各キューブレットを描画
    for cubelet in cube.cubelets:
        vertices, faces, face_mapping = _create_unit_cube(cubelet.position)
        colors = _get_face_colors(cubelet.rotation)

        for direction, face_idx in face_mapping.items():
            face_vertices = vertices[faces[face_idx]]
            poly = Poly3DCollection([face_vertices], alpha=1.0)
            poly.set_facecolor(colors[direction])
            poly.set_edgecolor('black')
            ax.add_collection3d(poly)

    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    plt.title('4×4×4 Cube Analysis Simulator')

    # 画像として保存
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
