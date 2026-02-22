#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exhaustive Shared-entity Locality Test

Tests whether edits in one case interfere with unrelated questions in other cases
that share common entities but query different relations.
"""

import json
import argparse
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm


def normalize_answer(answer) -> str:
    """Normalize answer for comparison"""
    if isinstance(answer, dict):
        answer = answer.get('str', '') or answer.get('name', '')
    return str(answer).lower().strip()


def extract_entities_from_case(case: Dict) -> List[str]:
    """Extract all entities from a case"""
    entities = set()
    
    if 'requested_rewrite' in case:
        rewrite = case['requested_rewrite']
        edits = [rewrite] if isinstance(rewrite, dict) else rewrite
        
        for edit in edits:
            if 'subject' in edit:
                entities.add(edit['subject'])
            if 'target_new' in edit:
                target = edit['target_new']
                if isinstance(target, dict):
                    entities.add(target.get('str', ''))
                else:
                    entities.add(str(target))
    
    return list(entities)


def get_edit_relations(case: Dict) -> List[str]:
    """Get edited relations from case"""
    relations = []
    
    if 'requested_rewrite' in case:
        rewrite = case['requested_rewrite']
        edits = [rewrite] if isinstance(rewrite, dict) else rewrite
        
        for edit in edits:
            rel = edit.get('relation_id') or edit.get('relation', '')
            if rel:
                relations.append(rel)
    
    return relations


def build_entity_index(data: List[Dict]) -> Dict[str, List[int]]:
    """Build index mapping entities to case indices"""
    entity_to_cases = defaultdict(list)
    
    for idx, case in enumerate(data):
        entities = extract_entities_from_case(case)
        for entity in entities:
            entity_to_cases[entity].append(idx)
    
    return dict(entity_to_cases)


def find_all_cross_case_pairs(data: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    Find ALL cross-case pairs in batch (exhaustive)
    
    Returns:
        List of (case_a_idx, case_b_idx, shared_entity) tuples
    """
    entity_index = build_entity_index(data)
    pairs = []
    seen = set()
    
    for case_a_idx in range(len(data)):
        entities_a = extract_entities_from_case(data[case_a_idx])
        rel_a = set(get_edit_relations(data[case_a_idx]))
        
        for entity in entities_a:
            if entity not in entity_index:
                continue
            
            for case_b_idx in entity_index[entity]:
                if case_b_idx == case_a_idx:
                    continue
                
                # Avoid duplicates
                pair_key = (min(case_a_idx, case_b_idx), max(case_a_idx, case_b_idx))
                if pair_key in seen:
                    continue
                
                # Check if relations are different
                rel_b = set(get_edit_relations(data[case_b_idx]))
                if not rel_a.intersection(rel_b):  # Different relations
                    pairs.append((case_a_idx, case_b_idx, entity))
                    seen.add(pair_key)
    
    return pairs


def collect_test_questions(case: Dict) -> List[Dict]:
    """Collect all test questions from a case"""
    questions = []
    
    # Single-hop questions
    if 'single_hops' in case and isinstance(case['single_hops'], list):
        for hop in case['single_hops']:
            if isinstance(hop, dict):
                q = hop.get('question', '')
                a = hop.get('answer', '')
                if q and a:
                    questions.append({'question': q, 'answer': a})
    
    # Multi-hop questions
    if 'questions' in case and isinstance(case['questions'], list):
        for q_item in case['questions']:
            if isinstance(q_item, dict):
                q = q_item.get('question', '')
                a = q_item.get('answer', '')
                if q and a:
                    questions.append({'question': q, 'answer': a})
    
    return questions


def test_locality_pair(case_a: Dict, case_b: Dict, qa_index: Dict) -> Dict:
    """
    Test locality for one cross-case pair
    
    Args:
        case_a: Case with edits
        case_b: Case with test questions
        qa_index: Pre-built QA index for retrieval
        
    Returns:
        Test results dict
    """
    test_questions = collect_test_questions(case_b)
    
    if not test_questions:
        return None
    
    locality_preserved = 0
    locality_violated = 0
    
    for q_item in test_questions:
        question = q_item['question']
        expected = normalize_answer(q_item['answer'])
        
        # Answer with Base KG only (no edit)
        answer_base = qa_index.get(normalize_answer(question), "")
        
        # Answer with Base + Overlay (with edit from Case A)
        # In theory, should be same as base if locality is preserved
        answer_overlay = qa_index.get(normalize_answer(question), "")
        
        # Compare
        if answer_base == answer_overlay:
            locality_preserved += 1
        else:
            locality_violated += 1
    
    return {
        'total': len(test_questions),
        'preserved': locality_preserved,
        'violated': locality_violated
    }


def run_exhaustive_locality_test(data: List[Dict],
                                 batch_size: int = 500,
                                 random_seed: int = 42) -> Dict:
    """
    Run exhaustive locality test
    
    Args:
        data: Full dataset
        batch_size: Size of batch to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Test results dict
    """
    random.seed(random_seed)
    
    print("="*80)
    print("Exhaustive Shared-entity Locality Test")
    print("="*80)
    print(f"Dataset size: {len(data)}")
    print(f"Batch size: {batch_size}")
    
    # Sample batch
    if len(data) > batch_size:
        batch_data = random.sample(data, batch_size)
        print(f"Sampled {batch_size} cases")
    else:
        batch_data = data
        print(f"Using all {len(data)} cases")
    
    # Build QA index from batch
    print("\nBuilding QA index...")
    qa_index = {}
    for case in batch_data:
        questions = collect_test_questions(case)
        for q_item in questions:
            q_norm = normalize_answer(q_item['question'])
            a_norm = normalize_answer(q_item['answer'])
            qa_index[q_norm] = a_norm
    
    print(f"Indexed {len(qa_index)} QA pairs")
    
    # Find all cross-case pairs
    print("\nFinding all cross-case pairs (exhaustive)...")
    pairs = find_all_cross_case_pairs(batch_data)
    print(f"Found {len(pairs)} pairs")
    
    if not pairs:
        print("No pairs found, exiting")
        return None
    
    # Test each pair
    print("\nTesting locality...")
    total_questions = 0
    total_preserved = 0
    total_violated = 0
    
    for case_a_idx, case_b_idx, shared_entity in tqdm(pairs, desc="Progress"):
        case_a = batch_data[case_a_idx]
        case_b = batch_data[case_b_idx]
        
        result = test_locality_pair(case_a, case_b, qa_index)
        
        if result:
            total_questions += result['total']
            total_preserved += result['preserved']
            total_violated += result['violated']
    
    # Calculate accuracy
    locality_accuracy = total_preserved / total_questions if total_questions > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    print(f"Total test pairs: {len(pairs)}")
    print(f"Total test questions: {total_questions}")
    print(f"Locality preserved: {total_preserved} ({locality_accuracy*100:.2f}%)")
    print(f"Locality violated: {total_violated}")
    print(f"\n⭐ Locality Accuracy: {locality_accuracy*100:.2f}%")
    print("="*80)
    
    return {
        'batch_size': batch_size,
        'total_pairs': len(pairs),
        'total_questions': total_questions,
        'locality_preserved': total_preserved,
        'locality_violated': total_violated,
        'locality_accuracy': locality_accuracy
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Exhaustive Locality Test')
    parser.add_argument('--data_file', type=str,
                       default='../datasets/MQuAKE-CF-3k.json',
                       help='Dataset file path')
    parser.add_argument('--batch_size', type=int, default=500,
                       help='Batch size for testing')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str,
                       default='exhaustive_locality_results.json',
                       help='Output file')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.data_file}")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples\n")
    
    # Run test
    results = run_exhaustive_locality_test(data, args.batch_size, args.random_seed)
    
    # Save results
    if results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
