#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import Counter


def load_cases(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_questions_from_cases(cases):
    """Flatten each case's ``questions`` list into a single list."""
    questions = []
    for item in cases:
        qs = item.get('questions') or []
        for q in qs:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())
    return questions


def build_cf_question_index(cf_cases):
    """Build a frequency index over all questions found in CF cases
    (top-level questions, single_hops, new_single_hops)."""
    counter = Counter()
    for item in cf_cases:
        for q in item.get('questions', []) or []:
            if isinstance(q, str) and q.strip():
                counter[q.strip()] += 1
        for hop in item.get('single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                counter[q] += 1
        for hop in item.get('new_single_hops', []) or []:
            q = (hop.get('question') or '').strip()
            if q:
                counter[q] += 1
    return counter


def count_overlaps(test_questions, cf_index):
    hits = [cf_index.get(q, 0) for q in test_questions]
    total = len(test_questions)
    num_nonzero = sum(1 for c in hits if c > 0)
    return total, num_nonzero, hits


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dataset_dir = os.path.join(base, 'datasets')

    cf_path = os.path.join(dataset_dir, 'MQuAKE-CF.json')
    t_path = os.path.join(dataset_dir, 'MQuAKE-T.json')
    cf3k_path = os.path.join(dataset_dir, 'MQuAKE-CF-3k.json')

    print('Loading datasets...')
    cf_cases = load_cases(cf_path)
    t_cases = load_cases(t_path)
    cf3k_cases = load_cases(cf3k_path)

    print('Building CF question index...')
    cf_index = build_cf_question_index(cf_cases)
    print(f'   CF index size (unique questions): {len(cf_index):,}')

    print('Extracting test questions (T)...')
    t_questions = extract_questions_from_cases(t_cases)
    print('Extracting test questions (CF-3k)...')
    cf3k_questions = extract_questions_from_cases(cf3k_cases)

    print('Counting overlaps for T...')
    t_total, t_nonzero, t_hits = count_overlaps(t_questions, cf_index)
    print(f'   T: total_questions={t_total:,}, overlapped_cases={t_nonzero:,} (question-level)')

    print('Counting overlaps for CF-3k...')
    c_total, c_nonzero, c_hits = count_overlaps(cf3k_questions, cf_index)
    print(f'   CF-3k: total_questions={c_total:,}, overlapped_cases={c_nonzero:,} (question-level)')

    print('Done.')


if __name__ == '__main__':
    main()
