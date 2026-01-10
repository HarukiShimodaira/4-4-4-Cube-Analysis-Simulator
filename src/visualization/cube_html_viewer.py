from typing import List
from ..core.cube4x4 import Cube4x4
from .cube_text_viewer import get_face_grid


def print_cube_state_html(cube: Cube4x4) -> str:
    """
    キューブの状態をHTML形式で表示します。
    WEB GUI用にANSIエスケープシーケンスを使わずに出力します。

    Args:
        cube (Cube4x4): 表示するキューブ
    
    Returns:
        str: HTML形式の文字列
    
    展開図の形式:
           U
        L  F  R  B
           D
    """
    # 色コードの文字マッピング（HTMLで表示用）
    COLOR_SYMBOLS = {
        'W': '⬜',  # 白
        'Y': '🟨',  # 黄
        'G': '🟩',  # 緑
        'B': '🟦',  # 青
        'R': '🟥',  # 赤
        'O': '🟧',  # オレンジ
        '.': '⬛'   # 不明
    }
    
    # 各面のグリッドを取得
    faces = {
        'U': get_face_grid(cube, 'U'),
        'D': get_face_grid(cube, 'D'),
        'F': get_face_grid(cube, 'F'),
        'R': get_face_grid(cube, 'R'),
        'B': get_face_grid(cube, 'B'),
        'L': get_face_grid(cube, 'L')
    }
    
    result = []
    result.append("現在のキューブ状態:\n")
    
    # 上面を表示
    result.append("       U")
    for row in faces['U']:
        symbols = [COLOR_SYMBOLS[c] for c in row]
        result.append("     " + " ".join(symbols))
    result.append("")
    
    # 中央の4面を表示
    result.append("   L      F      R      B")
    for i in range(4):
        row_symbols = []
        for face in ['L', 'F', 'R', 'B']:
            symbols = [COLOR_SYMBOLS[c] for c in faces[face][i]]
            row_symbols.extend(symbols)
            row_symbols.append(' ')
        result.append("  " + " ".join(row_symbols))
    result.append("")
    
    # 下面を表示
    result.append("       D")
    for row in faces['D']:
        symbols = [COLOR_SYMBOLS[c] for c in row]
        result.append("     " + " ".join(symbols))
    
    return "\n".join(result)
