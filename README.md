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
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">I edited</th>
      <th colspan="2">100 edited</th>
      <th colspan="2">All edited</th>
    </tr>
    <tr>
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
    </tr>
    <tr>
      <th colspan="7" style="text-align:center;">LLaMa 2-7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FT<sub>COT
      <td>22.30<td>-
      <td>2.13<td>-
      <td>OOM<td>-
    </tr>
    <tr>
      <td>FT</td>
      <td>28.20<td>7.30
      <td>2.37<td>0.03
      <td>OOM<td>OOM
    </tr>
    <tr>
      <td>ROME<sub>COT
      <td>11.17<td>–
      <td>2.87<td>–
      <td>2.77<td>–
    </tr>
    <tr>
      <td>ROME</td>
      <td>13.13<td>5.37
      <td>3.50<td>0.03
      <td>3.63<td>0.10
    </tr>
    <tr>
      <td>MEMIT<sub>COT
      <td>11.83<td>–
      <td>9.23<td>–
      <td>5.57<td>–
    </tr>
    <tr>
      <td>MEMIT</td>
      <td>14.97<td>6.43
      <td>9.40<td>2.47
      <td>2.30<td>0.37
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>33.57<td>9.90
      <td>20.00<td>10.07
      <td>17.33<td>9.90
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>44.13<td>30.90
      <td>37.33<td>27.83
      <td>32.83<td>23.87
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>66.80<td>63.67
      <td>58.50<td>55.37
      <td>48.30<td>43.90
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>78.03<td><strong>70.35
      <td><strong>78.93<td><strong>71.02
      <td><strong>78.38<td><strong>70.84
    </tr>
  </tbody>
</table>




### Main results (Evaluation Set MQuAKE-T)
### Ablation Studies


