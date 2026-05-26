#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json


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

    cf_path = os.path.join(ds_dir, 'MQuAKE-CF.json')
    t_path = os.path.join(ds_dir, 'MQuAKE-T.json')
    cf3k_path = os.path.join(ds_dir, 'MQuAKE-CF-3k.json')
    # Outputs:
    #   - top-level-question deduplication -> MQuAKE-CF_nonduplicate.json
    #   - top + single_hops deduplication -> MQuAKE-CF_answer_pool_nonduplicate.json
    out_top = os.path.join(ds_dir, 'MQuAKE-CF_nonduplicate.json')
    out_strict = os.path.join(ds_dir, 'MQuAKE-CF_answer_pool_nonduplicate.json')

    print('Loading datasets...')
    cf_cases = load_cases(cf_path)
    t_cases = load_cases(t_path)
    cf3k_cases = load_cases(cf3k_path)

    print('Building test question blacklist:')
    top_blacklist = extract_top_questions(t_cases) | extract_top_questions(cf3k_cases)
    strict_blacklist = set(top_blacklist)
    strict_blacklist |= extract_single_hop_questions(t_cases)
    strict_blacklist |= extract_single_hop_questions(cf3k_cases)
    print(f'   top_blacklist size: {len(top_blacklist):,}')
    print(f'   strict_blacklist size: {len(strict_blacklist):,}')

    # Version 1 (top): remove only top-level question duplicates
    print('Building CF_nonduplicate_top (remove top-level duplicates only)...')
    removed_top = kept_top = 0
    cases_top = []
    for item in cf_cases:
        new_item = dict(item)
        qs = [q for q in (item.get('questions') or []) if isinstance(q, str)]
        new_qs = []
        for q in qs:
            if q.strip() in top_blacklist:
                removed_top += 1
            else:
                new_qs.append(q)
                kept_top += 1
        new_item['questions'] = new_qs
        cases_top.append(new_item)
    with open(out_top, 'w', encoding='utf-8') as f:
        json.dump(cases_top, f, ensure_ascii=False, indent=2)
    print('Saved:', out_top)
    print(f'   top-level questions kept: {kept_top:,}, removed: {removed_top:,}')

    # Version 2 (strict): remove top-level + single_hops duplicates
    print('Building CF_nonduplicate_strict (remove top-level + single_hops duplicates)...')
    removed_top_s = kept_top_s = 0
    removed_hop_s = kept_hop_s = 0
    cases_strict = []
    for item in cf_cases:
        new_item = dict(item)
        # top-level questions
        qs = [q for q in (item.get('questions') or []) if isinstance(q, str)]
        new_qs = []
        for q in qs:
            if q.strip() in top_blacklist:
                removed_top_s += 1
            else:
                new_qs.append(q)
                kept_top_s += 1
        new_item['questions'] = new_qs
        # single_hops
        new_shops = []
        for hop in (item.get('single_hops') or []):
            q = (hop.get('question') or '').strip()
            if q and q in strict_blacklist:
                removed_hop_s += 1
            else:
                new_shops.append(hop)
                if q:
                    kept_hop_s += 1
        new_item['single_hops'] = new_shops
        # new_single_hops
        new_nshops = []
        for hop in (item.get('new_single_hops') or []):
            q = (hop.get('question') or '').strip()
            if q and q in strict_blacklist:
                removed_hop_s += 1
            else:
                new_nshops.append(hop)
                if q:
                    kept_hop_s += 1
        new_item['new_single_hops'] = new_nshops
        cases_strict.append(new_item)
    with open(out_strict, 'w', encoding='utf-8') as f:
        json.dump(cases_strict, f, ensure_ascii=False, indent=2)
    print('Saved:', out_strict)
    print(f'   top-level kept={kept_top_s:,}, removed={removed_top_s:,}; hops kept={kept_hop_s:,}, removed={removed_hop_s:,}')


if __name__ == '__main__':
    main()


