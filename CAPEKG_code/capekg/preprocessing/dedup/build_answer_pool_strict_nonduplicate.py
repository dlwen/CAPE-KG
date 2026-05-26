#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle


def load_cases(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_top_questions(cases):
    qs = set()
    for item in cases:
        for q in item.get('questions', []) or []:
            if isinstance(q, str) and q.strip():
                qs.add(q.strip())
    return qs


def extract_single_hop_questions(cases):
    qs = set()
    for item in cases:
        for hop in item.get('single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                qs.add(q)
        for hop in item.get('new_single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                qs.add(q)
    return qs


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    ds_dir = os.path.join(base, 'datasets')
    pools_dir = os.path.join(base, 'preprocessing', 'preprocessing', 'pools')

    t_path = os.path.join(ds_dir, 'MQuAKE-T.json')
    cf3k_path = os.path.join(ds_dir, 'MQuAKE-CF-3k.json')

    print('Loading test datasets (T, CF-3k)...')
    t_cases = load_cases(t_path)
    cf3k_cases = load_cases(cf3k_path)

    print('Building strict blacklist (top-level + single_hops)...')
    blacklist = set()
    # top-level questions
    blacklist |= extract_top_questions(t_cases)
    blacklist |= extract_top_questions(cf3k_cases)
    # single_hop questions
    blacklist |= extract_single_hop_questions(t_cases)
    blacklist |= extract_single_hop_questions(cf3k_cases)
    print(f'   blacklist size: {len(blacklist):,}')

    # Strictly deduplicate the answer_pool against the blacklist
    answer_pkl = os.path.join(pools_dir, 'answer_pool.pkl')
    with open(answer_pkl, 'rb') as f:
        answer_pool = pickle.load(f)

    kept, removed = [], []
    for item in answer_pool:
        q = (item.get('question') or '').strip()
        if q and q in blacklist:
            removed.append(item)
        else:
            kept.append(item)

    # Overwrite
    with open(os.path.join(pools_dir, 'answer_pool.pkl'), 'wb') as f:
        pickle.dump(kept, f)
    with open(os.path.join(pools_dir, 'answer_pool.json'), 'w', encoding='utf-8') as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print('answer_pool overwritten (strict nonduplicate):')
    print(f'   kept={len(kept):,}, removed={len(removed):,}')


if __name__ == '__main__':
    main()


