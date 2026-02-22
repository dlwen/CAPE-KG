#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
λ (Lambda) Parameter Ablation Study

Evaluates the impact of the λ parameter on outlier filtering.

λ controls the threshold for filtering outlier entities in statistical analysis.
Higher λ → more lenient filtering → more entities retained.

Usage:
    python lambda_ablation.py --lambda_values 0.5 1.0 1.5 2.0
"""

import json
import argparse
import sys
import os
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kedkg_main import KEDKG


def run_lambda_ablation(dataset: List[Dict],
                        lambda_values: List[float],
                        max_samples: int = 80) -> Dict:
    """
    Run λ parameter ablation study
    
    Args:
        dataset: Test dataset
        lambda_values: List of λ values to test
        max_samples: Number of samples per λ value
        
    Returns:
        Results dict with accuracy for each λ value
    """
    results = {}
    
    print(f"Running λ ablation study on {len(lambda_values)} values")
    print(f"Testing on {max_samples} samples per value\n")
    
    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        print(f"Testing λ = {lambda_val}")
        print(f"{'='*60}")
        
        # Initialize KEDKG with this λ value
        config = {
            'decomposer': 'gpt',
            'gpt_model': 'gpt-3.5-turbo-instruct',
            'tau': 0.5,  # Keep tau fixed at optimal value
            'lambda': lambda_val
        }
        
        kedkg = KEDKG(config)
        
        # Evaluate
        eval_results = kedkg.evaluate(dataset, max_samples=max_samples)
        
        # Store results
        results[lambda_val] = {
            'multi_hop_accuracy': eval_results['multi_hop_accuracy'],
            'hop_wise_accuracy': eval_results['hop_wise_accuracy'],
            'total_samples': eval_results['total_samples']
        }
        
        print(f"\nResults for λ = {lambda_val}:")
        print(f"  M-Acc: {eval_results['multi_hop_accuracy']:.4f}")
        print(f"  H-Acc: {eval_results['hop_wise_accuracy']:.4f}")
    
    return results


def print_summary(results: Dict):
    """Print summary table of results"""
    print("\n" + "="*60)
    print("λ Parameter Ablation Study - Summary")
    print("="*60)
    print(f"{'λ Value':<10} {'M-Acc':<12} {'H-Acc':<12} {'Samples':<10}")
    print("-"*60)
    
    for lambda_val, res in sorted(results.items()):
        print(f"{lambda_val:<10.1f} {res['multi_hop_accuracy']:<12.4f} "
              f"{res['hop_wise_accuracy']:<12.4f} {res['total_samples']:<10}")
    
    # Find best λ
    best_lambda = max(results.items(), key=lambda x: x[1]['multi_hop_accuracy'])
    print("-"*60)
    print(f"Best λ value: {best_lambda[0]} "
          f"(M-Acc: {best_lambda[1]['multi_hop_accuracy']:.4f})")
    print("="*60)


def save_results(results: Dict, output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='λ Parameter Ablation Study')
    parser.add_argument('--data_file', type=str,
                       default='../datasets/MQuAKE-CF-3k.json',
                       help='Dataset file path')
    parser.add_argument('--lambda_values', type=float, nargs='+',
                       default=[0.5, 1.0, 1.5, 2.0],
                       help='List of λ values to test')
    parser.add_argument('--max_samples', type=int, default=80,
                       help='Samples to test per λ value')
    parser.add_argument('--output', type=str,
                       default='lambda_ablation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.data_file}")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples\n")
    
    # Run ablation study
    results = run_lambda_ablation(data, args.lambda_values, args.max_samples)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
