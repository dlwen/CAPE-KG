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
<table>
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
  <tr>
    <td>CAPE-KG<_td><td>78.03<_td><td>70.35<_td><td>78.93<_td><td>71.02<_td><td>78.38<_td><td>70.84</td>
  </tr>
  <tr>
    <td>KeDKG<_td><td>66.80<_td><td>63.67<_td><td>58.50<_td><td>55.37<_td><td>48.30<_td><td>43.90</td>
  </tr>
</table>


### Main results (Evaluation Set MQuAKE-T)
### Ablation Studies


