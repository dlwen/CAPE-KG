#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
τ (Tau) Parameter Ablation Study

Evaluates the impact of the τ parameter on entity filtering.

τ controls the threshold for filtering entities based on statistical measures.
Higher τ → more aggressive filtering → fewer but more relevant entities.

Usage:
    python tau_ablation.py --tau_values 0.3 0.4 0.5 0.6 0.7
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


def run_tau_ablation(dataset: List[Dict],
                     tau_values: List[float],
                     max_samples: int = 80) -> Dict:
    """
    Run τ parameter ablation study
    
    Args:
        dataset: Test dataset
        tau_values: List of τ values to test
        max_samples: Number of samples per τ value
        
    Returns:
        Results dict with accuracy for each τ value
    """
    results = {}
    
    print(f"Running τ ablation study on {len(tau_values)} values")
    print(f"Testing on {max_samples} samples per value\n")
    
    for tau in tau_values:
        print(f"\n{'='*60}")
        print(f"Testing τ = {tau}")
        print(f"{'='*60}")
        
        # Initialize KEDKG with this τ value
        config = {
            'decomposer': 'gpt',
            'gpt_model': 'gpt-3.5-turbo-instruct',
            'tau': tau,
            'lambda': 1.5  # Keep lambda fixed
        }
        
        kedkg = KEDKG(config)
        
        # Evaluate
        eval_results = kedkg.evaluate(dataset, max_samples=max_samples)
        
        # Store results
        results[tau] = {
            'multi_hop_accuracy': eval_results['multi_hop_accuracy'],
            'hop_wise_accuracy': eval_results['hop_wise_accuracy'],
            'total_samples': eval_results['total_samples']
        }
        
        print(f"\nResults for τ = {tau}:")
        print(f"  M-Acc: {eval_results['multi_hop_accuracy']:.4f}")
        print(f"  H-Acc: {eval_results['hop_wise_accuracy']:.4f}")
    
    return results


def print_summary(results: Dict):
    """Print summary table of results"""
    print("\n" + "="*60)
    print("τ Parameter Ablation Study - Summary")
    print("="*60)
    print(f"{'τ Value':<10} {'M-Acc':<12} {'H-Acc':<12} {'Samples':<10}")
    print("-"*60)
    
    for tau, res in sorted(results.items()):
        print(f"{tau:<10.1f} {res['multi_hop_accuracy']:<12.4f} "
              f"{res['hop_wise_accuracy']:<12.4f} {res['total_samples']:<10}")
    
    # Find best τ
    best_tau = max(results.items(), key=lambda x: x[1]['multi_hop_accuracy'])
    print("-"*60)
    print(f"Best τ value: {best_tau[0]} "
          f"(M-Acc: {best_tau[1]['multi_hop_accuracy']:.4f})")
    print("="*60)


def save_results(results: Dict, output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='τ Parameter Ablation Study')
    parser.add_argument('--data_file', type=str,
                       default='../datasets/MQuAKE-CF-3k.json',
                       help='Dataset file path')
    parser.add_argument('--tau_values', type=float, nargs='+',
                       default=[0.3, 0.4, 0.5, 0.6, 0.7],
                       help='List of τ values to test')
    parser.add_argument('--max_samples', type=int, default=80,
                       help='Samples to test per τ value')
    parser.add_argument('--output', type=str,
                       default='tau_ablation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.data_file}")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples\n")
    
    # Run ablation study
    results = run_tau_ablation(data, args.tau_values, args.max_samples)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
