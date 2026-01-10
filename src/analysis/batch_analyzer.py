"""
Batch Analyzer Module
Runs multiple trials and collects statistics

Copyright (c) 2025 Haruki Shimodaira
Licensed under the MIT License - see LICENSE file for details
"""

from typing import List, Dict, Optional
import random
import time
from ..core.cube4x4 import Cube4x4
from .edge_tracker import EdgeTracker
from .position_analyzer import PositionAnalyzer
from .excel_exporter import ExcelExporter


class BatchAnalyzer:
    """
    Performs batch analysis of cube scrambles
    """
    
    # Operations including inner layers (for pair separation)
    ALL_OPERATIONS = [
        'R', 'L', 'U', 'D', 'F', 'B',
        'Rp', 'Lp', 'Up', 'Dp', 'Fp', 'Bp',
        'R2', 'L2', 'U2', 'D2', 'F2', 'B2',
        'r', 'l', 'u', 'd', 'f', 'b',
        'rp', 'lp', 'up', 'dp', 'fp', 'bp',
        'r2', 'l2', 'u2', 'd2', 'f2', 'b2'
    ]
    
    def __init__(self):
        """Initialize the batch analyzer"""
        self.tracker = EdgeTracker()
        self.analyzer = PositionAnalyzer()
        self.exporter = ExcelExporter()
    
    def run_single_trial(self, num_operations: int, seed: Optional[int] = None) -> Dict:
        """
        Run a single trial with random operations
        
        Args:
            num_operations: Number of random operations to perform
            seed: Random seed (optional)
            
        Returns:
            Dictionary with trial results
        """
        if seed is not None:
            random.seed(seed)
        
        # Create cube and perform operations
        cube = Cube4x4()
        operations = [random.choice(self.ALL_OPERATIONS) for _ in range(num_operations)]
        
        for op in operations:
            getattr(cube, op)()
        
        # Analyze
        edges = self.tracker.identify_edges(cube)
        pair_analyses = self.analyzer.analyze_all_pairs(edges)
        stats = self.analyzer.calculate_position_stats(edges, pair_analyses)
        patterns = self.analyzer.find_patterns(pair_analyses)
        
        return {
            'num_operations': num_operations,
            'operations': operations,
            'edges': edges,
            'pair_analyses': pair_analyses,
            'stats': stats,
            'patterns': patterns,
            # Summary data for Excel
            'separated_pairs': stats.separated_pairs,
            'pairs_on_same_face': stats.pairs_on_same_face,
            'avg_pair_distance': stats.average_pair_distance,
            'max_pair_distance': stats.max_pair_distance,
        }
    
    def run_trials(self, min_ops: int, max_ops: int, num_trials: int, 
                   verbose: bool = True, seed_base: Optional[int] = None) -> List[Dict]:
        """
        Run multiple trials with varying operation counts
        
        Args:
            min_ops: Minimum number of operations
            max_ops: Maximum number of operations
            num_trials: Number of trials to run
            verbose: Print progress
            seed_base: Base seed for reproducibility (None for random)
            
        Returns:
            List of trial results
        """
        results = []
        start_time = time.time()
        
        # Use current time as seed if not specified
        if seed_base is None:
            seed_base = int(time.time() * 1000) % 1000000
        
        if verbose:
            print(f"\nRunning {num_trials} trials with {min_ops}-{max_ops} operations...")
        
        for trial in range(num_trials):
            # Random operation count
            num_ops = random.randint(min_ops, max_ops)
            
            # Run trial
            result = self.run_single_trial(num_ops, seed=seed_base + trial)
            results.append(result)
            
            # Progress
            if verbose and (trial + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (trial + 1) / elapsed
                eta = (num_trials - trial - 1) / rate if rate > 0 else 0
                print(f"  Trial {trial + 1}/{num_trials} completed "
                      f"({elapsed:.1f}s elapsed, ETA: {eta:.1f}s)")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"Completed {num_trials} trials in {elapsed:.2f} seconds")
            print(f"Average: {elapsed/num_trials*1000:.1f}ms per trial")
        
        return results
    
    def export_results(self, results: List[Dict], filename: str, 
                      operations_range: tuple):
        """
        Export results to Excel
        
        Args:
            results: List of trial results
            filename: Output filename
            operations_range: (min_ops, max_ops)
        """
        self.exporter.export_trial_data(results, filename, operations_range)
    
    def analyze_and_export(self, min_ops: int = 15, max_ops: int = 30, 
                          num_trials: int = 100, output_file: str = "analysis.xlsx",
                          export_csv: bool = False):
        """
        Run analysis and export to Excel (and optionally CSV)
        
        Args:
            min_ops: Minimum operations per trial
            max_ops: Maximum operations per trial
            num_trials: Number of trials
            output_file: Output Excel filename
            export_csv: If True, also export raw data to CSV
        """
        print("=" * 70)
        print("4×4×4 Cube Analysis Simulator - Edge Pair Analysis")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Operations per trial: {min_ops}-{max_ops}")
        print(f"  Number of trials: {num_trials}")
        print(f"  Output file: {output_file}")
        
        # Run trials
        results = self.run_trials(min_ops, max_ops, num_trials, verbose=True)
        
        # Calculate summary statistics
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        
        avg_separated = sum(r['separated_pairs'] for r in results) / len(results)
        avg_distance = sum(r['avg_pair_distance'] for r in results) / len(results)
        avg_same_face = sum(r['pairs_on_same_face'] for r in results) / len(results)
        
        print(f"Average separated pairs: {avg_separated:.2f} / 12")
        print(f"Average pair distance: {avg_distance:.2f}")
        print(f"Average pairs on same face: {avg_same_face:.2f}")
        
        # Distance distribution
        distance_categories = {'near': 0, 'medium': 0, 'far': 0}
        for result in results:
            for category, count in result['patterns']['distance_distribution'].items():
                distance_categories[category] += count
        
        total = sum(distance_categories.values())
        print(f"\nOverall distance distribution:")
        for category, count in distance_categories.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {category:8s}: {count:4d} ({pct:5.1f}%)")
        
        # Export to Excel
        print("\n" + "=" * 70)
        print("Exporting to Excel...")
        print("=" * 70)
        self.export_results(results, output_file, (min_ops, max_ops))
        
        # Export to CSV if requested
        if export_csv:
            csv_filename = output_file.replace('.xlsx', '.csv')
            print("\n" + "=" * 70)
            print("Exporting to CSV...")
            print("=" * 70)
            exporter = ExcelExporter()
            exporter.export_csv(results, csv_filename)
        
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)
        print("=" * 70)
    
    def run_statistical_analysis(self, min_ops: int, max_ops: int, 
                                 trials_per_set: int = 100, 
                                 num_sets: int = 10,
                                 verbose: bool = True) -> List[Dict]:
        """
        Run statistical analysis across operation counts
        
        For each operation count from min_ops to max_ops:
        1. Run trials_per_set trials and calculate average statistics
        2. Repeat num_sets times
        3. Calculate mean and std dev across sets
        
        Args:
            min_ops: Minimum number of operations
            max_ops: Maximum number of operations
            trials_per_set: Number of trials per set
            num_sets: Number of sets to repeat
            verbose: Print progress
            
        Returns:
            List of statistical results per operation count
        """
        results = []
        total_ops = max_ops - min_ops + 1
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"統計分析を開始")
            print(f"{'='*70}")
            print(f"操作数範囲: {min_ops}-{max_ops} ({total_ops}個)")
            print(f"1セットあたりのトライアル数: {trials_per_set}")
            print(f"セット反復回数: {num_sets}")
            print(f"総トライアル数: {total_ops * trials_per_set * num_sets}")
            print(f"{'='*70}\n")
        
        for op_count in range(min_ops, max_ops + 1):
            set_results = []
            
            if verbose:
                print(f"操作数 {op_count}/{max_ops} の分析中...")
            
            # Run multiple sets for this operation count
            for set_idx in range(num_sets):
                # Run trials for this set
                trials = self.run_trials(op_count, op_count, trials_per_set, 
                                        verbose=False)
                
                # Calculate average statistics for this set
                avg_separated = sum(t['separated_pairs'] for t in trials) / len(trials)
                avg_distance = sum(t['avg_pair_distance'] for t in trials) / len(trials)
                avg_same_face = sum(t['pairs_on_same_face'] for t in trials) / len(trials)
                
                set_results.append({
                    'separated_pairs': avg_separated,
                    'avg_pair_distance': avg_distance,
                    'pairs_on_same_face': avg_same_face
                })
            
            # Calculate statistics across sets
            import numpy as np
            separated_values = [s['separated_pairs'] for s in set_results]
            distance_values = [s['avg_pair_distance'] for s in set_results]
            same_face_values = [s['pairs_on_same_face'] for s in set_results]
            
            results.append({
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
            
            if verbose:
                print(f"  分離ペア数: {results[-1]['separated_pairs_mean']:.2f} ± {results[-1]['separated_pairs_std']:.2f}")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"統計分析完了: {elapsed:.2f}秒")
            print(f"{'='*70}")
        
        return results
