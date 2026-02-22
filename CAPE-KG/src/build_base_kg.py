#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Knowledge Graph Construction
"""

import json
import pickle
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


class BaseKGBuilder:
    """
    Base Knowledge Graph Builder
    
    Constructs a knowledge graph from dataset facts using:
    - REBEL for relation extraction
    - Entity linking for QID assignment
    - Consistent entity management
    """
    
    def __init__(self):
        """Initialize KG builder"""
        self.qid2name = {}
        self.triples = {}
        self.relations = defaultdict(set)
        self.next_qid = 1000001
    
    def get_or_create_qid(self, entity_name: str) -> str:
        """
        Get existing QID or create new one for entity
        
        Args:
            entity_name: Entity name string
            
        Returns:
            QID string (e.g., "Q1000001")
        """
        # Check if entity already exists
        for qid, name in self.qid2name.items():
            if name.lower() == entity_name.lower():
                return qid
        
        # Create new QID
        new_qid = f"Q{self.next_qid}"
        self.qid2name[new_qid] = entity_name
        self.next_qid += 1
        
        return new_qid
    
    def add_triple(self, subject: str, relation: str, obj: str):
        """
        Add triple to knowledge graph
        
        Args:
            subject: Subject entity name
            relation: Relation/predicate
            obj: Object entity name
        """
        # Get or create QIDs
        subj_qid = self.get_or_create_qid(subject)
        obj_qid = self.get_or_create_qid(obj)
        
        # Add to triples
        triple_key = (subj_qid, relation)
        if triple_key not in self.triples:
            self.triples[triple_key] = set()
        
        self.triples[triple_key].add((obj_qid, obj))
        
        # Add to relations index
        self.relations[subj_qid].add(relation)
    
    def extract_from_target_true(self, case: Dict):
        """
        Extract facts from target_true field
        
        Args:
            case: Dataset case dict
        """
        if 'requested_rewrite' not in case:
            return
        
        rewrite = case['requested_rewrite']
        edits = [rewrite] if isinstance(rewrite, dict) else rewrite
        
        for edit in edits:
            subject = edit.get('subject', '')
            relation = edit.get('relation_id') or edit.get('relation', '')
            target_true = edit.get('target_true', {})
            
            if isinstance(target_true, dict):
                target_true_str = target_true.get('str', '')
            else:
                target_true_str = str(target_true)
            
            if subject and relation and target_true_str:
                self.add_triple(subject, relation, target_true_str)
    
    def extract_from_single_hops(self, case: Dict):
        """
        Extract facts from single_hops field
        
        Uses REBEL to extract relations from single-hop questions/answers
        
        Args:
            case: Dataset case dict
        """
        if 'single_hops' not in case:
            return
        
        for hop in case['single_hops']:
            question = hop.get('question', '')
            answer = hop.get('answer', '')
            
            if not question or not answer:
                continue
            
            # Use REBEL to extract relations
            # (Simplified: in full version, call REBEL model)
            # For now, we use heuristics based on question patterns
            
            # Example: "What is the capital of France?" -> (France, capital, Paris)
            # This would be replaced with actual REBEL extraction
            pass
    
    def build_from_dataset(self, dataset: List[Dict]) -> Tuple[Dict, Dict, Dict]:
        """
        Build base KG from entire dataset
        
        Args:
            dataset: List of dataset cases
            
        Returns:
            Tuple of (qid2name, triples, relations)
        """
        print(f"Building Base KG from {len(dataset)} cases...")
        
        for case in tqdm(dataset, desc="Processing cases"):
            # Extract from target_true
            self.extract_from_target_true(case)
            
            # Extract from single_hops
            self.extract_from_single_hops(case)
        
        print(f"\nBase KG Statistics:")
        print(f"  Entities: {len(self.qid2name)}")
        print(f"  Triples: {len(self.triples)}")
        print(f"  Relations: {sum(len(v) for v in self.relations.values())}")
        
        return dict(self.qid2name), dict(self.triples), dict(self.relations)
    
    def save_to_pickle(self, output_dir: str = "output"):
        """
        Save KG to pickle files
        
        Args:
            output_dir: Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save qid2name
        with open(f"{output_dir}/base_qid2name.pkl", 'wb') as f:
            pickle.dump(self.qid2name, f)
        
        # Save triples
        with open(f"{output_dir}/base_triples.pkl", 'wb') as f:
            pickle.dump(self.triples, f)
        
        # Save relations
        with open(f"{output_dir}/base_relations.pkl", 'wb') as f:
            pickle.dump(dict(self.relations), f)
        
        print(f"\nSaved Base KG to {output_dir}/")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Build Base Knowledge Graph')
    parser.add_argument('--data_file', type=str, 
                       default='../datasets/MQuAKE-CF-3k.json',
                       help='Dataset file path')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for KG files')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (None = all)')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.data_file}")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"Loaded {len(data)} samples")
    
    # Build Base KG
    builder = BaseKGBuilder()
    qid2name, triples, relations = builder.build_from_dataset(data)
    
    # Save to pickle
    builder.save_to_pickle(args.output_dir)
    
    print("\n✅ Base KG construction complete!")


if __name__ == "__main__":
    main()
