#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
from collections import defaultdict


def load_cases(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_cf_question_set(cf_cases):
    s = set()
    for item in cf_cases:
        for q in item.get('questions', []) or []:
            if isinstance(q, str) and q.strip():
                s.add(q.strip())
        for hop in item.get('single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                s.add(q)
        for hop in item.get('new_single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                s.add(q)
    return s


def filter_pool_items(pool_items, cf_question_set, question_key):
    """Drop pool entries whose question is in the CF blacklist
    (entry-level removal, not case-level)."""
    kept, removed = [], []
    for item in pool_items:
        q = item.get(question_key)
        if isinstance(q, str) and q.strip() in cf_question_set:
            removed.append(item)
        else:
            kept.append(item)
    return kept, removed


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dataset_dir = os.path.join(base, 'datasets')
    pools_dir = os.path.join(base, 'preprocessing', 'preprocessing', 'pools')
    out_dir = pools_dir  # overwrite the existing pool files in place

    # Blacklist = (questions in CF) - (questions kept in CF_nonduplicate);
    # i.e. those entries that the dedup pass removed because they collide
    # with the test sets.
    cf_path = os.path.join(dataset_dir, 'MQuAKE-CF.json')
    cf_nondup_path = os.path.join(dataset_dir, 'MQuAKE-CF_nonduplicate.json')

    cf_full = load_cases(cf_path)
    cf_full_q = build_cf_question_set(cf_full)

    if os.path.exists(cf_nondup_path):
        cf_nondup = load_cases(cf_nondup_path)
        cf_nondup_q = build_cf_question_set(cf_nondup)
        blacklist = cf_full_q - cf_nondup_q
        print('Using blacklist derived from CF - CF_nonduplicate')
    else:
        # If CF_nonduplicate has not been generated yet, do nothing.
        blacklist = set()
        print('CF_nonduplicate not found, blacklist is empty (no changes)')
    print(f'   blacklist size: {len(blacklist):,}')

    # Load existing pools
    decomp_pkl = os.path.join(pools_dir, 'decomposition_pool.pkl')
    answer_pkl = os.path.join(pools_dir, 'answer_pool.pkl')

    with open(decomp_pkl, 'rb') as f:
        decomposition_pool = pickle.load(f)
    with open(answer_pkl, 'rb') as f:
        answer_pool = pickle.load(f)

    # decomposition_pool uses key 'original_question',
    # answer_pool uses key 'question'.
    kept_decomp, removed_decomp = filter_pool_items(decomposition_pool, blacklist, 'original_question')
    kept_answer, removed_answer = filter_pool_items(answer_pool, blacklist, 'question')

    # Overwrite pickle pools
    with open(os.path.join(out_dir, 'decomposition_pool.pkl'), 'wb') as f:
        pickle.dump(kept_decomp, f)
    with open(os.path.join(out_dir, 'answer_pool.pkl'), 'wb') as f:
        pickle.dump(kept_answer, f)

    # Also write json versions for inspection
    with open(os.path.join(out_dir, 'decomposition_pool.json'), 'w', encoding='utf-8') as f:
        json.dump(kept_decomp, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'answer_pool.json'), 'w', encoding='utf-8') as f:
        json.dump(kept_answer, f, ensure_ascii=False, indent=2)

    print('Pools overwritten in:', out_dir)
    print(f'   decomposition: kept={len(kept_decomp):,}, removed={len(removed_decomp):,}')
    print(f'   answer       : kept={len(kept_answer):,}, removed={len(removed_answer):,}')


if __name__ == '__main__':
    main()


