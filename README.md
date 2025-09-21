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
      <th colspan="2">1 edited</th>
      <th colspan="2">100 edited</th>
      <th colspan="2">All edited</th>
    </tr>
    <tr>
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="7" style="text-align:center;">LLaMa 2-7B</th>
    </tr>
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


 <tr>
      <th colspan="7" style="text-align:center;">Vicuna-7B</th>
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>30.70<td>28.80
      <td>24.75<td>12.25
      <td>22.35<td>10.85
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>45.83<td>34.80
      <td>38.77<td>31.23
      <td>31.63<td>25.30
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>68.60<td>65.13
      <td>62.43<td>58.20
      <td>51.10<td>44.67
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>81.89<td><strong>78.74
      <td><strong>81.59<td><strong>78.85
      <td><strong>80.94<td><strong>78.49
    </tr>



 <tr>
      <th colspan="7" style="text-align:center;">GPT-3.5-turbo-instruct</th>
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>57.43<td>28.80
      <td>40.87<td>28.13
      <td>35.27<td>25.30
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>67.27<td>56.37
      <td>56.00<td>49.63
      <td>45.87<td>39.77
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>68.00<td>65.33
      <td>59.50<td>56.80
      <td>49.10<td>43.17
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>79.78<td><strong>73.65
      <td><strong>79.81<td><strong>73.75
      <td><strong>79.03<td><strong>73.00
    </tr>
  </tbody>
</table>

### Main results (Evaluation Set MQuAKE-T)
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">100 edited</th>
      <th colspan="2">All edited</th>
    </tr>
    <tr>
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan=“5” style="text-align:center;">LLaMa 2-7B</th>
    </tr>
    <tr>
      <td>FT<sub>COT
      <td>47.32<td>-
      <td>3.75<td>-
    </tr>
    <tr>
      <td>FT</td>
      <td>56.48<td>33.89
      <td>1.02<td>0.37
    </tr>
    <tr>
      <td>ROME<sub>COT
      <td>28.96<td>–
      <td>14.40<td>–
    </tr>
    <tr>
      <td>ROME</td>
      <td>24.89<td>17.99
      <td>1.71<td>0.32
    </tr>
    <tr>
      <td>MEMIT<sub>COT
      <td>36.88<td>–
      <td>31.58<td>–
    </tr>
    <tr>
      <td>MEMIT</td>
      <td>30.89<td>23.89
      <td>25.21<td>20.13
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>65.78<td>55.27
      <td>57.69<td>44.55
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>75.43<td>60.44
      <td>74.36<td>60.33
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>73.13<td>69.06
      <td>71.15<td>66.76
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>93.39<td><strong>84.52
      <td><strong>92.57<td><strong>83.82
    </tr>


 <tr>
      <th colspan=“5” style="text-align:center;">Vicuna-7B</th>
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>60.72<td>48.55
      <td>51.55<td>42.97
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>74.57<td>55.19
      <td>73.07<td>55.09
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>71.90<td>67.23
      <td>74.68<td>66.64
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>98.08<td><strong>89.14
      <td><strong>97.45<td><strong>88.20
    </tr>

 <tr>
      <th colspan=“5” style="text-align:center;">GPT-3.5-turbo-instruct</th>
    </tr>
    <tr>
      <td>MeLLo</td>
      <td>88.12<td>52.84
      <td>74.57<td>53.53
    </tr>
    <tr>
      <td>PokeMQA</td>
      <td>76.98<td>69.09
      <td>78.16<td>67.88
    </tr>
    <tr>
      <td>KeDKG</td>
      <td>78.75<td>76.18
      <td>77.19<td>73.77
    </tr>
    <tr>
      <td><strong>CAPE-KG
      <td><strong>97.11<td><strong>88.00
      <td><strong>96.72<td><strong>87.40
    </tr>
  </tbody>
</table>


### Ablation Studies

<table>
  <thead>
    <tr>
      <th colspan="3">Method</th>
      <th colspan="6">MQUAKE-CF-3K</th>
      <th colspan="4">MQUAKE-CF-T</th>
    </tr>
    <tr>
      <th>KG Construction</th>
      <th>Retrieval</th>
      <th>Update</th>
      <th colspan="2">1 edited</th>
      <th colspan="2">100 edited</th>
      <th colspan="2">All edited</th>
      <th colspan="2">1 edited</th>
      <th colspan="2">All edited</th>
    </tr>
    <tr>
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
      <th>M-Acc<th>H-Acc
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>–<td>✓<td>✓
      <td>75.00<td>69.00
      <td>74.67<td>69.04
      <td>75.67<td>69.00
      <td>96.00<td>86.60
      <td>95.80<td>86.60
    </tr>
    <tr>
      <td>✓<td>–<td>✓
      <td>53.33<td>43.00
      <td>51.33<td>46.67
      <td>51.93<td>46.00
      <td>87.33<td>81.67
      <td>87.67<td>82.00
    </tr>
    <tr>
      <td>✓<td>✓<_td><td>–<_td>
      <td>77.40<td>71.00</td>
      <td>74.58<td>68.14</td>
      <td>74.29<td>68.10</td>
      <td>96.60<td>87.10</td>
      <td>94.35<td>85.60</td>
    </tr>
    <tr>
      <td>✓<_td><td>✓<_td><td>✓</td>
      <td><strong>79.78<_strong><td><strong>73.65<_strong>
      <td><strong>79.81<_strong><td><strong>73.75<_strong>
      <td><strong>79.03<_strong><td><strong>73.00<_strong>
      <td><strong>97.11<_strong><td><strong>88.00<_strong>
      <td><strong>96.72<_strong><td><strong>87.40<_strong>
    </tr>
  </tbody>
</table>

