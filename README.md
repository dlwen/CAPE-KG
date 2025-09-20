# CAPE-KG
**Consistency-Aware Parameter-Preserving Knowledge Editing Framework for Multi-Hop Question Answering**
## Overview
CAPE-KG, Consistency-Aware Parameter-Preserving Editing with Knowledge Graphs, a novel consistency-aware framework for PPKE on MHQA. CAPE-KG ensures KG construction, update, and retrieval are always aligned with the requirements of the MHQA task, maintaining coherent reasoning over both unedited and edited knowledge.

![](README/method%20graph.png)


## Contributions
* The multi-layer KG architecture ensures the consistent knowledge boundary between factual and edited knowledge during PPKE.
* The case-isolated updating mechanisms with conflict arbitration guarantees edits are applied atomically and align with the case-isolated assumption.
* The edit-aware retrieval module with progressive strategies and edit-related retrieval routing ensures the retrieval process is always consistent with the intended KE.

## Datasets
Experiments are conducted on benchmark MQuAKE, includes MQuAKE-CF-3K and MQuAKE-T used for evaluation. Few-shot question decomposition demos are drawn from MQuAKE-CF after removing overlaps with the evaluation data, and the progressive retrieval
parameters are selected via grid search on it.

## Quantitative results
### Main results (Evaluation Set MQuAKE-CF-3k)
| Method   |        I edited        |       100 edited       |       All edited       |
|----------|------------------------|------------------------|------------------------|
|          | M-Acc   | H-Acc         | M-Acc   | H-Acc         | M-Acc   | H-Acc         |
| FT[^COT]| 22.30.  | -         | 2.13   | -         | OOM   | -         |
| FT    | 28.20   | 7.30        | 2.37   | 0.03         | OOM   | OOM        |
| ROME[^COT]| 11.17.  | -         | 2.87   | -         | 2.77   | -         |
| ROME   | 13.13   | 5.37        | 3.50   | 0.03         | 3.63   | 0.10        |
| MEMIT[^COT]| 11.83.  | -         | 9.23   | -         | 5.57  | -         |
| MEMIT   | 14.97   | 6.43        | 9.40   |2.47         | 2.30   |0.37       |
| MeLLo| 33.57  | 9.90        |20.00   | 10.07         | 17.33  | 9.90         |
| PokeMQA   | 44.13   | 30.90       | 37.33   |27.83         | 32.83   |23.87       |
| KeDKG    | 66.80   | 63.67         | 58.50   | 55.37         | 48.30   | 43.90         |
| CAPE-KG  | 78.03   | 70.35         | 78.93   | 71.02         | 78.38   | 70.84         |


<table>
  <thead>
    <tr>
      <th colspan="7">LLaMa 2-7B</th> <!-- centers by default for <th> -->
    </tr>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">I edited</th>
      <th colspan="2">100 edited</th>
      <th colspan="2">All edited</th>
    </tr>
    <tr>
      <th>M-Acc<_th><th>H-Acc<_th>
      <th>M-Acc<_th><th>H-Acc<_th>
      <th>M-Acc<_th><th>H-Acc<_th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FT<sup><a href="#fn-cot">CoT<_a><_sup></td>
      <td>22.30<_td><td>-<_td>
      <td>2.13<_td><td>-<_td>
      <td>OOM<_td><td>-<_td>
    </tr>
    <tr>
      <td>FT</td>
      <td>28.20<_td><td>7.30<_td>
      <td>2.37<_td><td>0.03<_td>
      <td>OOM<_td><td>OOM<_td>
    </tr>
    <tr>
      <td>ROME<sup><a href="#fn-cot">CoT<_a><_sup></td>
      <td>11.17<_td><td>-<_td>
      <td>2.87<_td><td>-<_td>
      <td>2.77<_td><td>-<_td>
    </tr>
    <tr>
      <td>ROME</td>
      <td>13.13<_td><td>5.37<_td>
      <td>3.50<_td><td>0.03<_td>
      <td>3.63<_td><td>0.10<_td>
    </tr>
    <tr>
      <td>MEMIT<sup><a href="#fn-cot">CoT<_a><_sup></td>
      <td>11.83<_td><td>-<_td>
      <td>9.23<_td><td>-<_td>
      <td>5.57<_td><td>-<_td>
    </tr>
    <tr>
      <td>MEMIT</td>
      <td>14.97<_td><td>6.43<_td>
      <td>9.40<_td><td>2.47<_td>
      <td>2.30<_td><td>0.37<_td>
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>33.57<_td><td>9.90<_td>
      <td>20.00<_td><td>10.07<_td>
      <td>17.33<_td><td>9.90<_td>
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>44.13<_td><td>30.90<_td>
      <td>37.33<_td><td>27.83<_td>
      <td>32.83<_td><td>23.87<_td>
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>66.80<_td><td>63.67<_td>
      <td>58.50<_td><td>55.37<_td>
      <td>48.30<_td><td>43.90<_td>
    </tr>
    <tr>
      <td><strong>CAPE-KG<_strong><_td>
      <td><strong>78.03<_strong><_td><td><strong>70.35<_strong><_td>
      <td><strong>78.93<_strong><_td><td><strong>71.02<_strong><_td>
      <td><strong>78.38<_strong><_td><td><strong>70.84<_strong><_td>
    </tr>
  </tbody>
</table>

### Main results (Evaluation Set MQuAKE-T)
### Ablation Studies


