#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4×4×4 Cube Analysis Simulator - Command Line Interface

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

import argparse
import readline  # コマンド履歴とライン編集のサポート

from src.core.cube4x4 import Cube4x4
from src.visualization.cube_viewer import visualize_cube
from src.visualization.cube_text_viewer import print_cube_state, get_face_grid
from src.analysis.cubelet_inspector import CubeletInspector


def count_colors(cube: Cube4x4):
    """キューブの各色の数をカウント"""
    colors = {'W': 0, 'Y': 0, 'G': 0, 'B': 0, 'R': 0, 'O': 0}
    for face in ['U', 'D', 'F', 'B', 'R', 'L']:
        face_grid = get_face_grid(cube, face)
        for row in face_grid:
            for color in row:
                colors[color] += 1
    return colors


def execute_move(cube: Cube4x4, move: str) -> bool:
    """
    指定された操作を実行します

    Args:
        cube: Cube4x4オブジェクト
        move: 実行する操作

    Returns:
        bool: 操作が成功したかどうか
    """
    try:
        # メソッド名のマッピング（'を"p"に変換）
        move_normalized = move.replace("'", "p")
        
        # Check if method exists
        if hasattr(cube, move_normalized):
            method = getattr(cube, move_normalized)
            method()
            return True
        else:
            print(f'Error: Invalid move "{move}"')
            return False
    except Exception as e:
        print(f'Error: {e}')
        return False


def interactive_mode(cube: Cube4x4):
    """
    インタラクティブモードを実行します
    """
    print("4×4×4 Cube Analysis Simulator")
    print("Copyright (c) 2025 Haruki Shimodaira")
    print("Licensed under the MIT License")
    print("\n利用可能な操作:")
    print("  基本回転: R, L, U, D, F, B (+ ', 2)")
    print("  内側層: r, l, u, d, f, b (+ ', 2)")
    print("  ワイド回転: Rw, Lw, Uw, Dw, Fw, Bw (+ ', 2)")
    print("  スライス: M, E, S (+ ', 2)")
    print("  全体回転: x, y, z (+ ', 2)")
    print("\nコマンド:")
    print("  show: 3Dビュー | text: テキスト表示 | reset: 初期化")
    print("  history: 操作履歴 | colors: 色の数を表示 | inspect: キューブレット検査 | help: ヘルプ | quit: 終了")
    print("\nヒント: スペースなしでも入力可 (例: RUR'U' または R U R' U')")
    
    # 初期状態を表示
    visualize_cube(cube)
    print("\n初期状態（テキスト表示）:")
    print_cube_state(cube)
    
    # 操作履歴
    move_history = []

    while True:
        try:
            command = input("\nコマンド > ").strip()
            
            # コメントを削除
            if '#' in command:
                command = command[:command.index('#')].strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'help':
                print("\n利用可能な操作:")
                print("  基本回転: R, L, U, D, F, B (+ ', 2)")
                print("    例: R (右面時計回り), R' (反時計回り), R2 (180度)")
                print("  内側層: r, l, u, d, f, b (+ ', 2)")
                print("    例: r (右内側層), r' (逆回転), r2 (180度)")
                print("  ワイド回転: Rw, Lw, Uw, Dw, Fw, Bw (+ ', 2)")
                print("    例: Rw (R面とr層を同時回転)")
                print("  スライス: M, E, S (+ ', 2)")
                print("    M: 中央縦層(L方向), E: 中央横層(D方向), S: 中央前後層(F方向)")
                print("  全体回転: x, y, z (+ ', 2)")
                print("    x: R方向, y: U方向, z: F方向")
                print("\nコマンド:")
                print("  show: 3Dビューを表示")
                print("  text: テキスト表示を更新")
                print("  reset: キューブを初期状態に戻す")
                print("  history: これまでの操作履歴を表示")
                print("  colors: 各色の数を表示")
                print("  inspect: キューブレット検査（座標・向き詳細）")
                print("  quit/exit: プログラムを終了")
                print("\nヒント:")
                print("  スペースなしでも認識: RUR'U' = R U R' U'")
                print("  括弧は無視されます: (R U R' U') x6")
                print("  シャープ記号以降はコメント: R U R' U' (メモ)")
                continue
            elif command.lower() == 'show':
                visualize_cube(cube)
                continue
            elif command.lower() == 'text':
                print_cube_state(cube)
                continue
            elif command.lower() == 'reset':
                cube = Cube4x4()
                move_history = []
                print("キューブを初期状態にリセットしました")
                print_cube_state(cube)
                visualize_cube(cube)
                continue
            elif command.lower() == 'history':
                if move_history:
                    print(f"\n操作履歴（{len(move_history)}手）:")
                    # 10手ごとに改行
                    for i in range(0, len(move_history), 10):
                        print("  " + " ".join(move_history[i:i + 10]))
                else:
                    print("\n操作履歴はありません")
                continue
            elif command.lower() == 'colors':
                colors = count_colors(cube)
                total = sum(colors.values())
                print(f"\n色の数:")
                print(f"  白(W): {colors['W']}個")
                print(f"  黄(Y): {colors['Y']}個")
                print(f"  緑(G): {colors['G']}個")
                print(f"  青(B): {colors['B']}個")
                print(f"  赤(R): {colors['R']}個")
                print(f"  橙(O): {colors['O']}個")
                print(f"  合計: {total}個 (期待値: 96個)")
                if total != 96:
                    print("  警告: 合計が96個ではありません！")
                elif any(count != 16 for count in colors.values()):
                    print("  警告: 各色16個になっていません！")
                else:
                    print("  色の数は正常です")
                continue
            elif command.lower() == 'inspect':
                # キューブレット検査モード
                inspector = CubeletInspector()
                print("\n=== キューブレット検査モード ===")
                print("1. 全キューブレット情報")
                print("2. エッジペア詳細検査")
                print("0. 戻る")
                
                choice = input("\n選択 > ").strip()
                
                if choice == '1':
                    states = inspector.get_all_cubelets_state(cube)
                    print(f"\n全キューブレット数: {len(states)}")
                    print(f"  コーナー: {sum(1 for s in states if s.type == 'corner')}個")
                    print(f"  エッジ  : {sum(1 for s in states if s.type == 'edge')}個")
                    print(f"  センター: {sum(1 for s in states if s.type == 'center')}個")
                    
                    print("\nタイプでフィルター (corner/edge/center) または all: ", end='')
                    filter_type = input().strip().lower()
                    
                    if filter_type == 'all':
                        display_states = states
                    elif filter_type in ['corner', 'edge', 'center']:
                        display_states = [s for s in states if s.type == filter_type]
                    else:
                        print("無効な選択です")
                        continue
                    
                    print(f"\n{'ID':>3} {'タイプ':>8} {'初期位置':^24} {'現在位置':^24} {'移動距離':>10} {'回転角':>8}")
                    print("-" * 95)
                    
                    for state in sorted(display_states, key=lambda x: x.id):
                        init_pos = f"({state.initial_position[0]:5.1f},{state.initial_position[1]:5.1f},{state.initial_position[2]:5.1f})"
                        curr_pos = f"({state.current_position[0]:5.1f},{state.current_position[1]:5.1f},{state.current_position[2]:5.1f})"
                        print(f"{state.id:3d} {state.type:>8} {init_pos:^24} {curr_pos:^24} {state.displacement:10.4f} {state.rotation_angle:8.2f}°")
                
                elif choice == '2':
                    pairs = inspector.get_edge_pair_list()
                    print(f"\n利用可能なエッジペア: {', '.join(pairs)}")
                    pair_id = input("\n検査するペアID (例: UF, UR, DB) > ").strip().upper()
                    
                    try:
                        inspection = inspector.inspect_edge_pair(cube, pair_id)
                        report = inspector.format_inspection_report(inspection)
                        print("\n" + report)
                    except ValueError as e:
                        print(f"エラー: {e}")
                
                continue
            
            # 括弧を削除
            command = command.replace('(', '').replace(')', '')
            
            # 複数の操作を連続して実行
            moves = []
            current_move = ""
            i = 0
            while i < len(command):
                char = command[i]
                
                if char.isspace():
                    # スペースの場合は現在の操作を確定
                    if current_move:
                        moves.append(current_move)
                        current_move = ""
                    i += 1
                elif char == "'":
                    # プライム記号の場合は前の操作に追加
                    current_move += char
                    i += 1
                elif char.isdigit() and current_move:
                    # 数字の場合（2のみサポート）
                    if char == '2':
                        current_move += char
                    i += 1
                elif char.isalpha():
                    # 文字の場合
                    if current_move and not (current_move[-1].isalpha() and char.islower()):
                        # 前の操作を確定（wなどの接尾辞以外）
                        moves.append(current_move)
                        current_move = char
                    else:
                        current_move += char
                    i += 1
                else:
                    i += 1
            
            if current_move:  # 最後の操作を追加
                moves.append(current_move)
            
            # 各操作を実行
            executed_moves = []
            for move in moves:
                if execute_move(cube, move):
                    executed_moves.append(move)
                    print(f'操作を実行: {move}')
                    # 操作のたびに状態を更新表示（テキストと3D）
                    print_cube_state(cube)
                    visualize_cube(cube, update=True)
            
            # 履歴に追加
            move_history.extend(executed_moves)
            
            # 操作後に色の数を自動チェック
            colors = count_colors(cube)
            total = sum(colors.values())
            if total != 96 or any(count != 16 for count in colors.values()):
                print(f"\n警告: 色の数に異常があります！")
                print(f"  白(W): {colors['W']}, 黄(Y): {colors['Y']}, 緑(G): {colors['G']}")
                print(f"  青(B): {colors['B']}, 赤(R): {colors['R']}, 橙(O): {colors['O']}")
                print(f"  合計: {total}個")
            
            # キューブの状態を検証
            valid, errors = cube.validate()
            if not valid:
                print("警告: キューブの状態が不正です")
                for error in errors:
                    print(f"  {error}")
        
        except KeyboardInterrupt:
            print("\nプログラムを終了します...")
            break
        except EOFError:
            print("\nプログラムを終了します...")
            break
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='4×4×4 Cube Analysis Simulator',
        epilog='Copyright (c) 2025 Haruki Shimodaira | Licensed under the MIT License'
    )
    parser.add_argument('moves', nargs='*', help='スペース区切りの操作列（例：R U R\' U\'）')
    parser.add_argument('--show', action='store_true', help='キューブの状態を表示')
    parser.add_argument('-i', '--interactive', action='store_true', help='インタラクティブモード')
    parser.add_argument('--analyze', action='store_true', help='エッジペア分析モード')
    parser.add_argument('--version', action='version', version='%(prog)s Release 1.10')
    parser.add_argument('--batch', type=int, metavar='N', help='バッチ分析（N回のトライアル）')
    parser.add_argument('--min-ops', type=int, default=20, help='バッチ分析の最小操作数（デフォルト: 20）')
    parser.add_argument('--max-ops', type=int, default=50, help='バッチ分析の最大操作数（デフォルト: 50）')
    parser.add_argument('--output', type=str, default='analysis_results.xlsx', help='出力Excelファイル名（デフォルト: analysis_results.xlsx）')
    parser.add_argument('--csv', action='store_true', help='CSV形式でも生データを出力')
    
    args = parser.parse_args()
    
    # Batch analysis mode
    if args.batch:
        from src.analysis.batch_analyzer import BatchAnalyzer
        analyzer = BatchAnalyzer()
        analyzer.analyze_and_export(
            min_ops=args.min_ops,
            max_ops=args.max_ops,
            num_trials=args.batch,
            output_file=args.output,
            export_csv=args.csv
        )
        return
    
    # キューブの初期化
    cube = Cube4x4()
    
    # エッジペア分析モード
    if args.analyze:
        from src.analysis.edge_tracker import EdgeTracker
        from src.analysis.position_analyzer import PositionAnalyzer
        
        # 操作を実行
        if args.moves:
            moves = []
            for move_str in args.moves:
                current_move = ""
                for char in move_str:
                    if char == "'":
                        current_move += char
                    elif current_move:
                        moves.append(current_move)
                        current_move = char
                    else:
                        current_move = char
                if current_move:
                    moves.append(current_move)
            
            for move in moves:
                if execute_move(cube, move):
                    print(f'Executed: {move}')
        
        # Analyze edge pairs
        tracker = EdgeTracker()
        edges = tracker.identify_edges(cube)
        analyzer = PositionAnalyzer()
        pair_analyses = analyzer.analyze_all_pairs(edges)
        stats = analyzer.calculate_position_stats(edges, pair_analyses)
        
        print("\nEdge Pair Analysis Results")
        
        print(f"\nOverall Statistics:")
        print(f"  Separated pairs: {stats.separated_pairs} / 12")
        print(f"  Pairs on same face: {stats.pairs_on_same_face} / 12")
        print(f"  Average pair distance: {stats.average_pair_distance:.2f}")
        print(f"  Maximum pair distance: {stats.max_pair_distance:.2f}")
        
        print(f"\nDistance Distribution:")
        near_count = sum(1 for p in pair_analyses if p.distance_category == 'near')
        medium_count = sum(1 for p in pair_analyses if p.distance_category == 'medium')
        far_count = sum(1 for p in pair_analyses if p.distance_category == 'far')
        print(f"  near (≤1.5)  : {near_count:3d} ({near_count/12*100:5.1f}%)")
        print(f"  medium (≤3.5): {medium_count:3d} ({medium_count/12*100:5.1f}%)")
        print(f"  far (>3.5)   : {far_count:3d} ({far_count/12*100:5.1f}%)")
        
        print(f"\nPair Distances:")
        for analysis in pair_analyses:
            status = "same face" if analysis.same_face else "separated"
            print(f"  {analysis.pair_id}: {analysis.current_distance:.2f} ({analysis.distance_category}) - {status}")
        return
    
    if args.interactive:
        interactive_mode(cube)
        return
    
    # 通常モード：コマンドライン引数の操作を実行
    if args.moves:
        # 入力された操作を個別に分割
        moves = []
        for move_str in args.moves:
            # 複数の操作が連結している場合は分割
            current_move = ""
            for char in move_str:
                if char == "'":  # プライム記号の場合は前の操作に追加
                    current_move += char
                elif current_move:  # 既に操作が始まっている場合
                    moves.append(current_move)
                    current_move = char
                else:  # 新しい操作の開始
                    current_move = char
            if current_move:  # 最後の操作を追加
                moves.append(current_move)
        
        # 各操作を実行
        for move in moves:
            if execute_move(cube, move):
                print(f'操作を実行: {move}')
    
    # キューブの状態を表示
    if args.show or not args.moves:
        visualize_cube(cube)


if __name__ == '__main__':
    main()
