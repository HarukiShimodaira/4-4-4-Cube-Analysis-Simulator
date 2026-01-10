#!/usr/bin/env python3
"""
Web-based GUI for 4×4×4 Cube Analysis Simulator
Browser-based interface to access all features

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License
"""

import json
import io
import base64
import os
import sys
import tempfile
import time

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

from src.core.cube4x4 import Cube4x4
from src.visualization.cube_text_viewer import print_cube_state
from src.visualization.cube_html_viewer import print_cube_state_html
from src.analysis.edge_tracker import EdgeTracker
from src.analysis.position_analyzer import PositionAnalyzer
from src.analysis.batch_analyzer import BatchAnalyzer
from src.analysis.excel_exporter import ExcelExporter
from src.analysis.cubelet_inspector import CubeletInspector
from src.analysis.random_data_collector import RandomDataCollector
from src.analysis.random_data_exporter import RandomDataExcelExporter
from src.io.json_handler import save_cube_state, load_cube_state

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cube-analysis-simulator-dev')
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=120,  # extend to avoid premature timeouts on long tasks
    ping_interval=25,
)

# グローバルキューブインスタンス
cube = Cube4x4()
history = []
stats_cancel_flag = False
random_data_cancel_flag = False


def emit_activity(tag: str, message: str, detail=None, level: str = 'info'):
    """Emit a lightweight activity log event to the UI."""
    socketio.emit('activity_log', {
        'tag': tag,
        'message': message,
        'detail': detail,
        'level': level,
        'timestamp': time.time(),
    })


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/api/cube/state', methods=['GET'])
def get_cube_state():
    """現在のキューブ状態を取得"""
    # HTML用のテキスト表現を取得
    text_state = print_cube_state_html(cube)
    
    # 各面のグリッドデータを取得
    from src.visualization.cube_text_viewer import get_face_grid
    faces = {
        'U': get_face_grid(cube, 'U'),
        'D': get_face_grid(cube, 'D'),
        'F': get_face_grid(cube, 'F'),
        'B': get_face_grid(cube, 'B'),
        'R': get_face_grid(cube, 'R'),
        'L': get_face_grid(cube, 'L')
    }
    
    # エッジペア情報
    tracker = EdgeTracker()
    edges = tracker.identify_edges(cube)
    edge_pairs = tracker.get_edge_pairs(edges)
    
    # ペアが完成しているかチェック（距離が近い）
    paired = []
    paired_edge_ids = set()
    
    for pair_id, (edge1, edge2) in edge_pairs.items():
        distance = tracker.calculate_pair_distance(edge1, edge2)
        if distance <= 1.5:  # 完成ペア判定の閾値（near カテゴリ基準）
            paired.append((edge1, edge2))
            paired_edge_ids.add(edge1.edge_id)
            paired_edge_ids.add(edge2.edge_id)
    
    unpaired = [e for e in edges if e.edge_id not in paired_edge_ids]
    
    # 簡易解析情報を作成（完成判定は全ペアが揃っているかで判断）
    is_solved = len(paired) == 12 and len(unpaired) == 0
    
    analysis = {
        'corners': {
            'correct_position': 8 if is_solved else 0,
            'correct_orientation': 8 if is_solved else 0
        },
        'edges': {
            'correct_position': 24 if is_solved else 0,
            'correct_orientation': 24 if is_solved else 0,
            'paired_count': len(paired)
        },
        'centers': {
            'correct_position': 24 if is_solved else 0
        },
        'overall': {
            'completion_percentage': 100.0 if is_solved else (len(paired) / 12.0 * 100.0)
        }
    }
    
    return jsonify({
        'text_state': text_state,
        'faces': faces,
        'analysis': analysis,
        'edges': {
            'paired_count': len(paired),
            'unpaired_count': len(unpaired),
            'paired': [{'pos1': p1.edge_id, 'pos2': p2.edge_id} for p1, p2 in paired],
            'unpaired': [e.edge_id for e in unpaired]
        },
        'history': history,
        'is_solved': is_solved
    })


@app.route('/api/cube/execute', methods=['POST'])
def execute_operation():
    """操作を実行"""
    data = request.get_json()
    operation = data.get('operation', '')
    
    if not operation:
        return jsonify({'error': 'No operation provided'}), 400
    
    try:
        # 操作列をパース（スペースで分割）
        moves = operation.split()
        executed_count = 0
        
        for move in moves:
            # 'を"p"に変換
            move_normalized = move.replace("'", "p")
            
            # メソッドが存在するかチェック
            if hasattr(cube, move_normalized):
                method = getattr(cube, move_normalized)
                method()
                executed_count += 1
                # 履歴には個別の操作を記録（逆操作生成のため）
                history.append(move_normalized)
            else:
                return jsonify({'error': f'Invalid move: {move}'}), 400
        return jsonify({
            'success': True,
            'message': f"Executed: {operation}",
            'move_count': executed_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cube/reset', methods=['POST'])
def reset_cube():
    """キューブをリセット"""
    global cube, history
    cube = Cube4x4()
    history.clear()
    return jsonify({'success': True, 'message': 'Cube reset'})


@app.route('/api/cube/shuffle', methods=['POST'])
def shuffle_cube_api():
    """キューブをシャッフル"""
    global cube, history
    import random
    
    data = request.get_json() or {}
    moves_count = data.get('moves', 20)
    allowed_types = data.get('allowed_types', ['outer', 'slice'])  # デフォルトは全種類
    
    # 操作タイプ別の定義
    move_types = {
        'outer': [  # 外層のみ
            'R', 'Rp', 'R2', 'L', 'Lp', 'L2', 
            'U', 'Up', 'U2', 'D', 'Dp', 'D2',
            'F', 'Fp', 'F2', 'B', 'Bp', 'B2'
        ],
        'slice': [  # 中間層のみ
            'r', 'rp', 'r2', 'l', 'lp', 'l2',
            'u', 'up', 'u2', 'd', 'dp', 'd2',
            'f', 'fp', 'f2', 'b', 'bp', 'b2'
        ],
        'basic': [  # 基本操作のみ（90度回転）
            'R', 'Rp', 'L', 'Lp', 
            'U', 'Up', 'D', 'Dp',
            'F', 'Fp', 'B', 'Bp'
        ],
        'outer_90': [  # 外層90度のみ
            'R', 'Rp', 'L', 'Lp', 
            'U', 'Up', 'D', 'Dp',
            'F', 'Fp', 'B', 'Bp'
        ],
        'outer_180': [  # 外層180度のみ
            'R2', 'L2', 'U2', 'D2', 'F2', 'B2'
        ],
        'slice_90': [  # 中間層90度のみ
            'r', 'rp', 'l', 'lp',
            'u', 'up', 'd', 'dp',
            'f', 'fp', 'b', 'bp'
        ],
        'slice_180': [  # 中間層180度のみ
            'r2', 'l2', 'u2', 'd2', 'f2', 'b2'
        ]
    }
    
    # 許可された操作タイプから操作リストを構築
    all_moves = []
    for move_type in allowed_types:
        if move_type in move_types:
            all_moves.extend(move_types[move_type])
    
    # 重複を削除
    all_moves = list(set(all_moves))
    
    if not all_moves:
        return jsonify({
            'success': False,
            'error': '有効な操作タイプが指定されていません'
        }), 400
    
    shuffle_sequence = []
    for _ in range(moves_count):
        move = random.choice(all_moves)
        if hasattr(cube, move):
            method = getattr(cube, move)
            method()
            shuffle_sequence.append(move)
            history.append(move)
    
    return jsonify({
        'success': True,
        'message': f'Shuffled with {len(shuffle_sequence)} moves',
        'sequence': ' '.join(shuffle_sequence),
        'moves_count': len(shuffle_sequence),
        'allowed_types': allowed_types,
        'available_moves': len(all_moves)
    })


@app.route('/api/cube/save', methods=['GET'])
def save_cube():
    """キューブ状態をJSON形式でダウンロード"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    try:
        save_cube_state(cube, temp_file.name)
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name='cube_state.json',
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cube/load', methods=['POST'])
def load_cube():
    """JSONファイルからキューブ状態を読み込み"""
    global cube
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False)
        file.save(temp_file.name)
        temp_file.close()
        
        cube = load_cube_state(temp_file.name)
        os.unlink(temp_file.name)
        
        return jsonify({'success': True, 'message': 'Cube loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cube/visualize', methods=['GET'])
def visualize_cube():
    """3D可視化画像を生成"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # バックエンドを非GUI版に
        from src.visualization.cube_viewer import visualize_cube
        import matplotlib.pyplot as plt
        
        # クエリパラメータから視点を取得
        elev = request.args.get('elev', type=float, default=30)
        azim = request.args.get('azim', type=float, default=-45)
        
        # 画像をメモリに生成
        fig = visualize_cube(cube, show=False, elev=elev, azim=azim)
        
        # PNG形式でメモリに保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Base64エンコード
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/operations/list', methods=['GET'])
def list_operations():
    """利用可能な操作リストを取得"""
    operations = {
        'basic': ['R', 'L', 'U', 'D', 'F', 'B'],
        'prime': ["R'", "L'", "U'", "D'", "F'", "B'"],
        'double': ['R2', 'L2', 'U2', 'D2', 'F2', 'B2'],
        'wide': ['Rw', 'Lw', 'Uw', 'Dw', 'Fw', 'Bw'],
        'wide_prime': ["Rw'", "Lw'", "Uw'", "Dw'", "Fw'", "Bw'"],
        'wide_double': ['Rw2', 'Lw2', 'Uw2', 'Dw2', 'Fw2', 'Bw2'],
        'slice': ['r', 'l', 'u', 'd', 'f', 'b'],
        'slice_prime': ["r'", "l'", "u'", "d'", "f'", "b'"],
        'slice_double': ['r2', 'l2', 'u2', 'd2', 'f2', 'b2']
    }
    return jsonify(operations)


@app.route('/api/cube/colors', methods=['GET'])
def count_colors():
    """キューブの各色の数をカウント"""
    from src.visualization.cube_text_viewer import get_face_grid
    
    colors = {'W': 0, 'Y': 0, 'G': 0, 'B': 0, 'R': 0, 'O': 0}
    for face in ['U', 'D', 'F', 'B', 'R', 'L']:
        face_grid = get_face_grid(cube, face)
        for row in face_grid:
            for color in row:
                colors[color] += 1
    
    total = sum(colors.values())
    is_valid = total == 96 and all(count == 16 for count in colors.values())
    
    return jsonify({
        'colors': colors,
        'total': total,
        'is_valid': is_valid,
        'expected_total': 96,
        'expected_per_color': 16
    })


@app.route('/api/cube/validate', methods=['GET'])
def validate_cube():
    """キューブの状態を検証"""
    valid, errors = cube.validate()
    return jsonify({
        'is_valid': valid,
        'errors': errors
    })


@app.route('/api/cube/edge_analysis', methods=['GET'])
def detailed_edge_analysis():
    """詳細なエッジペア分析"""
    tracker = EdgeTracker()
    edges = tracker.identify_edges(cube)
    analyzer = PositionAnalyzer()
    pair_analyses = analyzer.analyze_all_pairs(edges)
    stats = analyzer.calculate_position_stats(edges, pair_analyses)
    
    # 距離カテゴリ別にカウント
    near_count = sum(1 for p in pair_analyses if p.distance_category == 'near')
    medium_count = sum(1 for p in pair_analyses if p.distance_category == 'medium')
    far_count = sum(1 for p in pair_analyses if p.distance_category == 'far')
    
    # ペア情報を詳細化
    pair_details = []
    for analysis in pair_analyses:
        pair_details.append({
            'pair_id': analysis.pair_id,
            'distance': round(analysis.current_distance, 2),
            'category': analysis.distance_category,
            'same_face': analysis.same_face,
            'status': 'same face' if analysis.same_face else 'separated'
        })
    
    return jsonify({
        'overall_stats': {
            'separated_pairs': stats.separated_pairs,
            'pairs_on_same_face': stats.pairs_on_same_face,
            'average_distance': round(stats.average_pair_distance, 2),
            'max_distance': round(stats.max_pair_distance, 2)
        },
        'distance_distribution': {
            'near': {'count': near_count, 'percentage': round(near_count / 12 * 100, 1)},
            'medium': {'count': medium_count, 'percentage': round(medium_count / 12 * 100, 1)},
            'far': {'count': far_count, 'percentage': round(far_count / 12 * 100, 1)}
        },
        'pair_details': pair_details
    })


@app.route('/api/batch/analyze', methods=['POST'])
def run_batch_analysis():
    """
    バッチ分析を実行
    リクエスト: {num_trials, min_ops, max_ops}
    """
    try:
        data = request.get_json()
        num_trials = data.get('num_trials', 100)
        min_ops = data.get('min_ops', 15)
        max_ops = data.get('max_ops', 30)
        
        # バリデーション
        if num_trials < 1 or num_trials > 1000:
            return jsonify({'error': 'トライアル数は1-1000の範囲で指定してください'}), 400
        if min_ops < 1 or max_ops > 200:
            return jsonify({'error': '操作数は1-200の範囲で指定してください'}), 400
        if min_ops > max_ops:
            return jsonify({'error': '最小操作数は最大操作数以下にしてください'}), 400
        
        # バッチ分析実行
        analyzer = BatchAnalyzer()
        emit_activity('batch', 'バッチ分析開始', {'min_ops': min_ops, 'max_ops': max_ops, 'trials': num_trials})
        results = analyzer.run_trials(min_ops, max_ops, num_trials, verbose=False)
        
        # 統計計算
        avg_separated = sum(r['separated_pairs'] for r in results) / len(results)
        avg_distance = sum(r['avg_pair_distance'] for r in results) / len(results)
        avg_same_face = sum(r['pairs_on_same_face'] for r in results) / len(results)
        
        # 距離分布
        distance_categories = {'near': 0, 'medium': 0, 'far': 0}
        for result in results:
            for category, count in result['patterns']['distance_distribution'].items():
                distance_categories[category] += count
        
        total_dist = sum(distance_categories.values())
        distance_percentages = {
            category: (count / total_dist * 100 if total_dist > 0 else 0)
            for category, count in distance_categories.items()
        }
        
        return jsonify({
            'success': True,
            'num_trials': num_trials,
            'operations_range': [min_ops, max_ops],
            'statistics': {
                'avg_separated_pairs': round(avg_separated, 2),
                'avg_pair_distance': round(avg_distance, 2),
                'avg_same_face': round(avg_same_face, 2),
                'distance_distribution': distance_categories,
                'distance_percentages': {
                    k: round(v, 1) for k, v in distance_percentages.items()
                }
            },
            'results_id': id(results)  # メモリ内の結果を一時保存
        })
        emit_activity('batch', '完了', {
            'min_ops': min_ops,
            'max_ops': max_ops,
            'trials': num_trials,
            'avg_pair_distance': round(avg_distance, 2)
        })
    
    except Exception as e:
        emit_activity('batch', 'エラー', {'error': str(e)}, level='error')
        return jsonify({'error': str(e)}), 500


# 一時的な結果保存用（本番環境ではRedisなど使用）
_batch_results_cache = {}


@app.route('/api/batch/export/<format>', methods=['POST'])
def export_batch_results(format):
    """
    バッチ分析結果をExcel/CSV形式でエクスポート
    format: 'excel' or 'csv'
    """
    try:
        data = request.get_json()
        num_trials = data.get('num_trials', 100)
        min_ops = data.get('min_ops', 15)
        max_ops = data.get('max_ops', 30)
        
        # バッチ分析実行（新規）
        analyzer = BatchAnalyzer()
        results = analyzer.run_trials(min_ops, max_ops, num_trials, verbose=False)
        
        # 一時ファイルに出力
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                         suffix='.xlsx' if format == 'excel' else '.csv') as tmp:
            tmp_path = tmp.name
        
        exporter = ExcelExporter()
        
        if format == 'excel':
            emit_activity('batch', 'Excel生成中: ワークブック構築', {'trials': num_trials, 'ops': f'{min_ops}-{max_ops}'})
            exporter.export_trial_data(results, tmp_path, (min_ops, max_ops))
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f'cube_analysis_{num_trials}trials_{min_ops}-{max_ops}ops.xlsx'
        else:  # csv
            emit_activity('batch', 'CSV生成中: 出力', {'trials': num_trials, 'ops': f'{min_ops}-{max_ops}'})
            exporter.export_csv(results, tmp_path)
            mimetype = 'text/csv'
            filename = f'cube_analysis_{num_trials}trials_{min_ops}-{max_ops}ops.csv'
        
        # ファイル送信後に削除
        emit_activity('batch', 'ファイル送信準備', {'filename': filename})
        response = send_file(
            tmp_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
        # クリーンアップは後で（Flask処理後）
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        
        return response
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('start_stats_analysis_progress')
def handle_stats_with_progress(data):
    """
    統計分析を実行（進捗通知付き）- threading mode
    """
    global stats_cancel_flag
    stats_cancel_flag = False
    
    def run_analysis():
        global stats_cancel_flag
        try:
            min_ops = data.get('min_ops', 1)
            max_ops = data.get('max_ops', 100)
            trials_per_set = data.get('trials_per_set', 100)
            num_sets = data.get('num_sets', 10)
            
            total_ops = max_ops - min_ops + 1
            analyzer = BatchAnalyzer()
            stats_results = []
            emit_activity('stats', '統計分析開始', {
                'min_ops': min_ops,
                'max_ops': max_ops,
                'trials_per_set': trials_per_set,
                'num_sets': num_sets,
            })
            
            for idx, op_count in enumerate(range(min_ops, max_ops + 1), 1):
                # キャンセルチェック
                if stats_cancel_flag:
                    socketio.emit('stats_error', {'error': 'キャンセルされました'})
                    emit_activity('stats', 'キャンセル', None, level='warning')
                    return
                
                # 進捗通知
                socketio.emit('stats_progress', {
                    'current': idx,
                    'total': total_ops,
                    'operation_count': op_count,
                    'percent': int(idx / total_ops * 100)
                })
                emit_activity('stats', f'ops {op_count} 分析中', {
                    'current': idx,
                    'total': total_ops,
                    'percent': int(idx / total_ops * 100),
                })
                socketio.sleep(0)  # yield control
                
                # 統計計算
                set_results = []
                for set_idx in range(num_sets):
                    # キャンセルチェック
                    if stats_cancel_flag:
                        socketio.emit('stats_error', {'error': 'キャンセルされました'})
                        return
                    
                    trials = analyzer.run_trials(op_count, op_count, trials_per_set, verbose=False)
                    avg_separated = sum(t['separated_pairs'] for t in trials) / len(trials)
                    avg_distance = sum(t['avg_pair_distance'] for t in trials) / len(trials)
                    avg_same_face = sum(t['pairs_on_same_face'] for t in trials) / len(trials)
                    
                    set_results.append({
                        'separated_pairs': avg_separated,
                        'avg_pair_distance': avg_distance,
                        'pairs_on_same_face': avg_same_face
                    })
                    socketio.sleep(0)  # yield control
                
                import numpy as np
                separated_values = [s['separated_pairs'] for s in set_results]
                distance_values = [s['avg_pair_distance'] for s in set_results]
                same_face_values = [s['pairs_on_same_face'] for s in set_results]
                
                stats_results.append({
                    'num_operations': op_count,
                    'separated_pairs_mean': np.mean(separated_values),
                    'separated_pairs_std': np.std(separated_values, ddof=1) if len(separated_values) > 1 else 0,
                    'avg_pair_distance_mean': np.mean(distance_values),
                    'avg_pair_distance_std': np.std(distance_values, ddof=1) if len(distance_values) > 1 else 0,
                    'pairs_on_same_face_mean': np.mean(same_face_values),
                    'pairs_on_same_face_std': np.std(same_face_values, ddof=1) if len(same_face_values) > 1 else 0,
                    'num_sets': num_sets,
                    'trials_per_set': trials_per_set
                })
            
            # Excel出力
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.xlsx') as tmp:
                tmp_path = tmp.name
            
            exporter = ExcelExporter()
            emit_activity('stats', 'Excel生成中: ワークブック構築', {
                'ops_range': f"{min_ops}-{max_ops}",
                'trials_per_set': trials_per_set,
                'num_sets': num_sets
            })
            exporter.export_statistical_analysis(
                stats_results, tmp_path, min_ops, max_ops, trials_per_set, num_sets
            )
            emit_activity('stats', 'Excel生成中: ファイル出力完了', {'path': tmp_path})
            
            # Base64エンコード
            filename = f'cube_stats_{min_ops}-{max_ops}ops_{trials_per_set}x{num_sets}.xlsx'
            with open(tmp_path, 'rb') as f:
                file_bytes = f.read()
            emit_activity('stats', 'Excel生成中: エンコード', {'size_bytes': len(file_bytes)})
            file_data = base64.b64encode(file_bytes).decode('utf-8')
            
            socketio.emit('stats_complete', {
                'filename': filename,
                'data': file_data
            })
            emit_activity('stats', '完了', {'filename': filename})
            
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        
        except Exception as e:
            socketio.emit('stats_error', {'error': str(e)})
            emit_activity('stats', 'エラー', {'error': str(e)}, level='error')
    
    socketio.start_background_task(run_analysis)


@socketio.on('cancel_stats_analysis')
def handle_cancel_stats():
    """統計分析をキャンセル"""
    global stats_cancel_flag
    stats_cancel_flag = True


@socketio.on('start_random_data_collection')
def handle_random_data_collection(data):
    """
    ランダムデータ収集を実行（進捗通知付き）
    """
    global random_data_cancel_flag
    random_data_cancel_flag = False
    
    def run_collection():
        global random_data_cancel_flag
        try:
            num_operations = data.get('num_operations', 100)
            seed = data.get('seed', None)  # None = random seed
            allowed_operations = data.get('allowed_operations', None)
            start_time = time.time()

            if allowed_operations is not None:
                if not isinstance(allowed_operations, list):
                    raise ValueError('allowed_operations は配列で指定してください')
                allowed_operations = [op for op in allowed_operations if isinstance(op, str)]
                if len(allowed_operations) == 0:
                    raise ValueError('使用する操作を1つ以上選択してください')
            
            collector = RandomDataCollector(seed=seed, allowed_operations=allowed_operations)
            emit_activity('random', 'データ収集開始', {
                'ops': num_operations,
                'seed': seed,
                'allowed_ops': collector.allowed_operations,
                'allowed_ops_count': len(collector.allowed_operations),
            })
            progress_gate = {'next': 0}
            
            # Progress callback
            def progress_callback(current, total, message):
                if random_data_cancel_flag:
                    raise Exception('キャンセルされました')
                elapsed = time.time() - start_time
                ops_per_sec = current / elapsed if elapsed > 0 else 0
                eta = (total - current) / ops_per_sec if ops_per_sec > 0 else None
                # Emit activity sparsely so the feed stays readable
                if current >= progress_gate['next'] or current == total:
                    step = max(1, total // 20)
                    progress_gate['next'] = min(total, current + step)
                    emit_activity('random', f'{current}/{total} {message}', {
                        'current': current,
                        'total': total,
                        'ops_per_sec': ops_per_sec,
                        'eta': eta,
                    })
                
                socketio.emit('random_data_progress', {
                    'current': current,
                    'total': total,
                    'message': message,
                    'percent': int(current / total * 100) if total > 0 else 0,
                    'elapsed_sec': elapsed,
                    'ops_per_sec': ops_per_sec,
                    'eta_sec': eta
                })
                socketio.sleep(0)  # yield control
            
            # Collect data
            collected_data = collector.collect_data(num_operations, progress_callback)
            
            # Notify Excel generation started
            socketio.emit('random_data_progress', {
                'current': num_operations,
                'total': num_operations,
                'message': 'Excelファイル生成中...',
                'percent': 100,
                'elapsed_sec': time.time() - start_time,
                'ops_per_sec': None,
                'eta_sec': None
            })
            emit_activity('random', 'Excel生成中: ワークブック構築', {'ops': num_operations})
            socketio.sleep(0)
            
            # Export to Excel
            excel_bytes = RandomDataExcelExporter.export_to_excel(collected_data)
            emit_activity('random', 'Excel生成中: バイト出力完了', {'size_bytes': len(excel_bytes)})
            socketio.sleep(0)
            
            # Base64 encode
            filename = f'random_data_{num_operations}ops_seed{collected_data["seed"]}.xlsx'
            file_data = base64.b64encode(excel_bytes).decode('utf-8')
            emit_activity('random', 'Excel生成中: エンコード完了', {'filename': filename})
            
            socketio.emit('random_data_complete', {
                'filename': filename,
                'data': file_data,
                'seed': collected_data['seed'],
                'num_operations': num_operations
            })
            emit_activity('random', '完了', {'filename': filename, 'ops': num_operations})
        
        except Exception as e:
            socketio.emit('random_data_error', {'error': str(e)})
            emit_activity('random', 'エラー', {'error': str(e)}, level='error')
    
    socketio.start_background_task(run_collection)


@socketio.on('cancel_random_data_collection')
def handle_cancel_random_data():
    """ランダムデータ収集をキャンセル"""
    global random_data_cancel_flag
    random_data_cancel_flag = True


@app.route('/api/server/restart', methods=['POST'])
def restart_server():
    """サーバーを再起動"""
    import signal
    import sys
    import subprocess
    
    def shutdown_and_restart():
        # 新しいプロセスで自分自身を起動
        subprocess.Popen([sys.executable] + sys.argv)
        # 現在のプロセスを終了
        os.kill(os.getpid(), signal.SIGTERM)
    
    # 1秒後にシャットダウンして再起動
    from threading import Timer
    Timer(1.0, shutdown_and_restart).start()
    
    return jsonify({'success': True, 'message': 'Server restarting...'})


@app.route('/api/server/kill', methods=['POST'])
def kill_server():
    """サーバーを停止"""
    import signal
    
    def shutdown():
        # 現在のプロセスを終了
        os.kill(os.getpid(), signal.SIGTERM)
    
    # 1秒後にシャットダウン
    from threading import Timer
    Timer(1.0, shutdown).start()
    
    return jsonify({'success': True, 'message': 'Server shutting down...'})


# ========== WebSocket Events ==========

@socketio.on('connect')
def handle_connect():
    """クライアント接続時"""
    print(f'Client connected')
    emit('connection_response', {'data': 'Connected to 4×4×4 Cube Analysis Simulator'})


@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時"""
    print(f'Client disconnected')


@socketio.on('execute_operation')
def handle_socket_operation(data):
    """WebSocket経由で操作を実行"""
    global cube, history
    print(f"[WebSocket] execute_operation received: {data}")
    operation = data.get('operation', '')
    
    if not operation:
        print("[WebSocket] No operation provided")
        emit('error', {'message': 'No operation provided'})
        return
    
    try:
        # 操作列をパース
        moves = operation.split()
        executed_count = 0
        
        for move in moves:
            move_normalized = move.replace("'", "p")
            
            if hasattr(cube, move_normalized):
                method = getattr(cube, move_normalized)
                method()
                executed_count += 1
                history.append(move_normalized)
                print(f"[WebSocket] Executed move: {move}")
            else:
                print(f"[WebSocket] Invalid move: {move}")
                emit('error', {'message': f'Invalid move: {move}'})
                return
        
        # 全クライアントに状態をブロードキャスト
        cube_state = get_current_state()
        print(f"[WebSocket] Broadcasting cube_updated to all clients")
        socketio.emit('cube_updated', cube_state, to=None)
        
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('request_state')
def handle_state_request():
    """現在の状態を要求"""
    cube_state = get_current_state()
    emit('cube_state', cube_state)


@socketio.on('reset_cube')
def handle_socket_reset():
    """WebSocket経由でリセット"""
    global cube, history
    cube = Cube4x4()
    history.clear()
    
    cube_state = get_current_state()
    socketio.emit('cube_updated', cube_state, to=None)


@socketio.on('shuffle_cube')
def handle_socket_shuffle(data):
    """WebSocket経由でシャッフル"""
    global cube, history
    import random
    
    moves_count = data.get('moves', 20)
    print(f"[WebSocket] Shuffling with {moves_count} moves")
    
    # ランダム操作のリスト（外層 + 中間層）
    all_moves = [
        'R', 'Rp', 'R2', 'L', 'Lp', 'L2', 
        'U', 'Up', 'U2', 'D', 'Dp', 'D2',
        'F', 'Fp', 'F2', 'B', 'Bp', 'B2',
        'r', 'rp', 'r2', 'l', 'lp', 'l2',
        'u', 'up', 'u2', 'd', 'dp', 'd2',
        'f', 'fp', 'f2', 'b', 'bp', 'b2'
    ]
    
    shuffle_sequence = []
    for _ in range(moves_count):
        move = random.choice(all_moves)
        if hasattr(cube, move):
            method = getattr(cube, move)
            method()
            shuffle_sequence.append(move)
            history.append(move)
    
    print(f"[WebSocket] Shuffle sequence: {' '.join(shuffle_sequence)}")
    
    cube_state = get_current_state()
    socketio.emit('cube_updated', cube_state, to=None)


def get_current_state():
    """現在のキューブ状態を取得（内部用）"""
    from src.visualization.cube_text_viewer import get_face_grid
    
    # HTML用のテキスト表現
    text_state = print_cube_state_html(cube)
    
    # 各面のグリッドデータ
    faces = {
        'U': get_face_grid(cube, 'U'),
        'D': get_face_grid(cube, 'D'),
        'F': get_face_grid(cube, 'F'),
        'B': get_face_grid(cube, 'B'),
        'R': get_face_grid(cube, 'R'),
        'L': get_face_grid(cube, 'L')
    }
    
    # 3Dデータ（各キューブレットの位置と色）
    # cube_viewer.pyの_get_face_colors()を完全にコピー
    cubelets_3d = []
    
    # cube_viewer.pyのCOLORS定義
    VIEWER_COLORS = {
        'U': 'W',  # 上面（白）
        'D': 'Y',  # 下面（黄）
        'F': 'G',  # 前面（緑）
        'B': 'B',  # 背面（青）
        'R': 'R',  # 右面（赤）
        'L': 'O'   # 左面（オレンジ）
    }
    
    # cube_viewer.pyの_get_face_colors()を完全コピー
    def get_face_colors_from_viewer(position, rotation):
        """
        cube_viewer.pyの_get_face_colors()と完全に同じロジック
        キューブレットの位置と回転から、外側から見える面の色を決定
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
                colors[face] = VIEWER_COLORS[best_match]
        
        return colors
    
    # 各キューブレットの色を計算
    for cubelet in cube.cubelets:
        rotation_matrix = cubelet.rotation.m.tolist()
        
        # cube_viewer.pyと同じロジックで色を取得
        viewer_colors = get_face_colors_from_viewer(cubelet.position, cubelet.rotation)
        
        # Three.jsのキー名にマッピング (R->right, L->left, U->top, D->bottom, F->front, B->back)
        colors = {
            'right': viewer_colors.get('R'),
            'left': viewer_colors.get('L'),
            'top': viewer_colors.get('U'),
            'bottom': viewer_colors.get('D'),
            'front': viewer_colors.get('F'),
            'back': viewer_colors.get('B')
        }
        
        cubelet_data = {
            'position': {
                'x': cubelet.position.x,
                'y': cubelet.position.y,
                'z': cubelet.position.z
            },
            'rotation': rotation_matrix,
            'initial_position': {
                'x': cubelet.initial_position.x,
                'y': cubelet.initial_position.y,
                'z': cubelet.initial_position.z
            },
            'colors': colors
        }
        cubelets_3d.append(cubelet_data)
    
    # エッジペア情報
    tracker = EdgeTracker()
    edges = tracker.identify_edges(cube)
    edge_pairs = tracker.get_edge_pairs(edges)
    
    paired = []
    paired_edge_ids = set()
    
    for pair_id, (edge1, edge2) in edge_pairs.items():
        distance = tracker.calculate_pair_distance(edge1, edge2)
        if distance < 1.5:
            paired.append((edge1, edge2))
            paired_edge_ids.add(edge1.edge_id)
            paired_edge_ids.add(edge2.edge_id)
    
    unpaired = [e for e in edges if e.edge_id not in paired_edge_ids]
    is_solved = len(paired) == 12 and len(unpaired) == 0
    
    # 解析情報を作成
    analysis = {
        'corners': {
            'correct_position': 8 if is_solved else 0,
            'correct_orientation': 8 if is_solved else 0
        },
        'edges': {
            'correct_position': 24 if is_solved else 0,
            'correct_orientation': 24 if is_solved else 0,
            'paired_count': len(paired)
        },
        'centers': {
            'correct_position': 24 if is_solved else 0
        },
        'overall': {
            'completion_percentage': 100.0 if is_solved else (len(paired) / 12.0 * 100.0)
        }
    }
    
    return {
        'text_state': text_state,
        'faces': faces,
        'cubelets_3d': cubelets_3d,
        'analysis': analysis,
        'edges': {
            'paired_count': len(paired),
            'unpaired_count': len(unpaired),
            'paired': [{'pos1': p1.edge_id, 'pos2': p2.edge_id} for p1, p2 in paired],
            'unpaired': [e.edge_id for e in unpaired]
        },
        'history': history,
        'is_solved': is_solved
    }


# ========== Interactive Mode API ==========

@app.route('/api/interactive/menu', methods=['GET'])
def get_interactive_menu():
    """インタラクティブモードのメニュー構造を取得"""
    menu = {
        'basic': [
            {'id': '1', 'label': 'キューブの状態を表示', 'action': 'show_state'},
            {'id': '2', 'label': '操作を実行', 'action': 'execute_single'},
            {'id': '3', 'label': '連続操作を実行', 'action': 'execute_sequence'},
            {'id': '4', 'label': 'キューブをリセット', 'action': 'reset'}
        ],
        'display': [
            {'id': '5', 'label': '操作履歴を表示', 'action': 'show_history'},
            {'id': '6', 'label': '3D表示を開く', 'action': 'show_3d'}
        ],
        'analysis': [
            {'id': '7', 'label': 'エッジペア解析', 'action': 'analyze_edges'},
            {'id': '8', 'label': 'バッチ解析実行', 'action': 'batch_analysis'},
            {'id': '9', 'label': '位置情報詳細表示', 'action': 'position_info'}
        ],
        'file': [
            {'id': '10', 'label': 'キューブを保存 (JSON)', 'action': 'save'},
            {'id': '11', 'label': 'キューブを読み込み (JSON)', 'action': 'load'},
            {'id': '12', 'label': 'Excel出力', 'action': 'export_excel'}
        ],
        'help': [
            {'id': '13', 'label': '全操作リスト表示', 'action': 'operations_list'},
            {'id': '14', 'label': 'ヘルプ', 'action': 'help'}
        ]
    }
    return jsonify(menu)


@app.route('/api/interactive/operations_list', methods=['GET'])
def get_all_operations_list():
    """全操作リストを取得"""
    ops = {
        "基本回転 (90度)": ["R", "L", "U", "D", "F", "B"],
        "反時計回り (90度)": ["R'", "L'", "U'", "D'", "F'", "B'"],
        "180度回転": ["R2", "L2", "U2", "D2", "F2", "B2"],
        "Wide回転 (外側2層)": ["Rw", "Lw", "Uw", "Dw", "Fw", "Bw"],
        "Wide反時計": ["Rw'", "Lw'", "Uw'", "Dw'", "Fw'", "Bw'"],
        "Wide 180度": ["Rw2", "Lw2", "Uw2", "Dw2", "Fw2", "Bw2"],
        "Slice (内側2層)": ["r", "l", "u", "d", "f", "b"],
        "Slice反時計": ["r'", "l'", "u'", "d'", "f'", "b'"],
        "Slice 180度": ["r2", "l2", "u2", "d2", "f2", "b2"]
    }
    return jsonify({'operations': ops, 'total_count': 72})


@app.route('/api/interactive/help', methods=['GET'])
def get_help_text():
    """ヘルプテキストを取得"""
    help_text = """
4×4×4 Cube Analysis Simulator

【基本的な使い方】
1. メニューから機能を選択
2. 操作を実行してキューブを回転
3. 状態を確認しながらパズルを解く

【操作記法】
- R, L, U, D, F, B: 基本的な90度回転
- ' (プライム): 反時計回り (例: R')
- 2: 180度回転 (例: R2)
- w: Wide回転 (外側2層、例: Rw)
- 小文字: Slice回転 (内側2層、例: r)

【解析機能】
- エッジペア解析: 完成しているエッジペアを確認
- バッチ解析: 複数の操作列を一括で解析
- 位置情報: 各ピースの正確性を確認

【ファイル操作】
- JSON形式でキューブ状態を保存/読み込み
- Excel形式で解析結果を出力

詳細はドキュメントを参照してください。
    """
    return jsonify({'help': help_text})


@app.route('/api/inspector/cubelets', methods=['GET'])
def get_all_cubelets():
    """全キューブレットの座標・向き情報を取得"""
    try:
        inspector = CubeletInspector()
        states = inspector.get_all_cubelets_state(cube)
        
        # 型別にグループ化
        by_type = {
            'corner': [],
            'edge': [],
            'center': []
        }
        
        for state in states:
            by_type[state.type].append({
                'id': state.id,
                'initial_position': state.initial_position,
                'current_position': state.current_position,
                'displacement': round(state.displacement, 4),
                'rotation_angle': round(state.rotation_angle, 2),
                'rotation_axis': state.rotation_axis,
                'rotation_matrix': state.rotation_matrix
            })
        
        return jsonify({
            'success': True,
            'total_count': len(states),
            'by_type': {
                'corner': {'count': len(by_type['corner']), 'cubelets': by_type['corner']},
                'edge': {'count': len(by_type['edge']), 'cubelets': by_type['edge']},
                'center': {'count': len(by_type['center']), 'cubelets': by_type['center']}
            },
            'all_cubelets': [{
                'id': s.id,
                'type': s.type,
                'initial_position': s.initial_position,
                'current_position': s.current_position,
                'displacement': round(s.displacement, 4),
                'rotation_angle': round(s.rotation_angle, 2),
                'rotation_axis': s.rotation_axis
            } for s in states]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inspector/edge_pairs', methods=['GET'])
def get_edge_pair_list():
    """利用可能なエッジペアのリストを取得"""
    try:
        inspector = CubeletInspector()
        pairs = inspector.get_edge_pair_list()
        
        return jsonify({
            'success': True,
            'pairs': pairs,
            'count': len(pairs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inspector/edge_pair/<pair_id>', methods=['GET'])
def inspect_specific_edge_pair(pair_id):
    """
    特定のエッジペアを詳細に調査
    
    Args:
        pair_id: ペアID (例: UF, UR, DB など)
    """
    try:
        inspector = CubeletInspector()
        inspection = inspector.inspect_edge_pair(cube, pair_id)
        
        # テキストレポートも生成
        text_report = inspector.format_inspection_report(inspection)
        
        return jsonify({
            'success': True,
            'pair_id': inspection.pair_id,
            'edge1': {
                'id': inspection.edge1_id,
                'initial_position': inspection.edge1_initial_pos,
                'current_position': inspection.edge1_current_pos,
                'displacement': round(inspection.edge1_displacement, 4),
                'rotation_angle': round(inspection.edge1_rotation_angle, 2),
                'rotation_axis': inspection.edge1_rotation_axis
            },
            'edge2': {
                'id': inspection.edge2_id,
                'initial_position': inspection.edge2_initial_pos,
                'current_position': inspection.edge2_current_pos,
                'displacement': round(inspection.edge2_displacement, 4),
                'rotation_angle': round(inspection.edge2_rotation_angle, 2),
                'rotation_axis': inspection.edge2_rotation_axis
            },
            'pair_info': {
                'initial_distance': round(inspection.initial_distance, 4),
                'current_distance': round(inspection.current_distance, 4),
                'distance_change': round(inspection.distance_change, 4),
                'same_face': inspection.same_face,
                'shared_face': inspection.shared_face,
                'relative_rotation_angle': round(inspection.relative_rotation_angle, 2)
            },
            'text_report': text_report
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_local_ip():
    """ローカルIPアドレスを取得"""
    import socket
    try:
        # ダミー接続でローカルIPを取得
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except OSError:
        return "127.0.0.1"


def generate_qr_code(url):
    """QRコードを生成してターミナルに表示"""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except ImportError:
        return False


def main():
    """メイン関数"""
    port = 8888
    local_ip = get_local_ip()
    
    print("\n" + "=" * 60)
    print("4×4×4 Cube Analysis Simulator - Web GUI")
    print("=" * 60)
    print("\nStarting web server with WebSocket support...")
    print()
    print("アクセスURL:")
    print(f"  ローカル:     http://localhost:{port}")
    print(f"  ネットワーク: http://{local_ip}:{port}")
    print()
    print("スマホからアクセスする場合:")
    print(f"  1. 同じWiFiネットワークに接続")
    print(f"  2. ブラウザで http://{local_ip}:{port} にアクセス")
    print()
    
    # QRコード生成
    url = f"http://{local_ip}:{port}"
    if generate_qr_code(url):
        print("\n上記QRコードをスマホでスキャンしてください")
        print()
    else:
        print("QRコード表示: pip install qrcode でインストール可能")
        print()
    
    print("終了するには Ctrl+C を押してください")
    print("=" * 60 + "\n")
    
    # Werkzeugのログを抑制
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
