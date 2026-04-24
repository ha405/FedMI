# CKA Analysis: Mechanistic Divergence Results

These results track the mechanistic divergence between IID and Non-IID federated training runs using a three-tier CKA analysis:
1. **Latent Total**: Similarity of the full model on the full dataset.
2. **Latent Per-Class (Unmasked)**: Similarity of raw model representations for each specific class label.
3. **Circuit Per-Class (Masked)**: Similarity of extracted mechanistic sub-networks.

## CIFAR10 ResNet-10

### Global Model Similarity (IID vs Non-IID)
Tracks the drift between the global models of two different training runs over 10 rounds.

| Round | Latent Total | Circuit Avg | Lat_0 | Lat_1 | Lat_2 | Lat_3 | Lat_4 | Circ_0 | Circ_1 | Circ_2 | Circ_3 | Circ_4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.5497 | 0.0000 | 0.2223 | 0.1764 | 0.1198 | 0.0465 | 0.0463 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **2** | 0.0841 | 0.2000 | 0.0768 | 0.0381 | 0.0617 | 0.0544 | 0.0655 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| **3** | 0.2157 | 0.0000 | 0.1441 | 0.1691 | 0.0827 | 0.0445 | 0.0435 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **4** | 0.3273 | 0.0057 | 0.1835 | 0.2312 | 0.1741 | 0.1126 | 0.1030 | 0.0000 | 0.0283 | 0.0000 | 0.0000 | 0.0000 |
| **5** | 0.3254 | 0.2395 | 0.1693 | 0.2185 | 0.1587 | 0.1086 | 0.2458 | 0.0000 | 0.1976 | 0.0000 | 0.0000 | 1.0000 |
| **6** | 0.3804 | 0.2306 | 0.2503 | 0.2797 | 0.2829 | 0.1539 | 0.2136 | 0.0000 | 0.1532 | 1.0000 | 0.0000 | 0.0000 |
| **7** | 0.4694 | 0.1352 | 0.3129 | 0.3861 | 0.3664 | 0.1690 | 0.3015 | 0.0000 | 0.5772 | 0.0000 | 0.0987 | 0.0000 |
| **8** | 0.4673 | 0.3219 | 0.3149 | 0.3588 | 0.3509 | 0.1934 | 0.2815 | 0.1384 | 0.4710 | 0.0000 | 0.0000 | 1.0000 |
| **9** | 0.4110 | 0.2878 | 0.2503 | 0.2679 | 0.2999 | 0.1721 | 0.2437 | 0.0000 | 0.4392 | 0.0000 | 0.0000 | 1.0000 |
| **10** | **0.4033** | **0.5116** | **0.3377** | **0.3170** | **0.3403** | **0.1673** | **0.3072** | **0.0000** | **0.5580** | **1.0000** | **0.0000** | **1.0000** |

### Inter-Client Similarity (Non-IID, Round 10)
Compares the similarity between local models of different clients at the final round.

| Comparison | Latent Total | Circuit Avg | Lat_0 | Lat_1 | Lat_2 | Lat_3 | Lat_4 | Circ_0 | Circ_1 | Circ_2 | Circ_3 | Circ_4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 0 vs 1** | 1.0000 | 0.1033 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5165 | 0.0000 | 0.0000 | 0.0000 |
| **Client 0 vs 2** | 1.0000 | 0.4048 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0242 | 1.0000 | 0.0000 | 0.0000 |
| **Client 1 vs 2** | 1.0000 | 0.0158 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0544 | 0.0000 | 0.0247 | 0.0000 |

> [!IMPORTANT]
> **Lat_X** represents raw model similarity (unmasked). **Circ_X** represents mechanistic circuit similarity (masked). A Circuit CKA of **1.0000** indicates perfect functional preservation, while **0.0000** indicates disjoint specialization.

---

## MNIST ResNet (Long-Horizon Study)

### Global Progression (IID vs Non-IID)
| Round | Latent Total | Circuit Avg | Lat_0 | Lat_1 | Lat_2 | Lat_3 | Lat_4 | Circ_0 | Circ_1 | Circ_2 | Circ_3 | Circ_4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.5381 | 0.0000 | 0.5958 | 0.7664 | 0.3399 | 0.2422 | 0.4630 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **2** | 0.5505 | 0.2000 | 0.3689 | 0.4165 | 0.2269 | 0.2215 | 0.3830 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **5** | 0.5533 | 0.4000 | 0.2408 | 0.5054 | 0.1388 | 0.1835 | 0.2731 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **8** | 0.7340 | 0.4000 | 0.2778 | 0.6025 | 0.1926 | 0.2690 | 0.2749 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **10** | **0.6628** | **0.4051** | **0.1992** | **0.5597** | **0.1537** | **0.1887** | **0.3158** | **1.0000** | **0.0254** | **0.0000** | **1.0000** | **0.0000** |

### Long-Horizon Convergence (Round 10 vs Round 50)
Compares how the models evolve over the extended training horizon.

| Comparison | Latent Total | Circuit Avg | Lat_0 | Lat_1 | Lat_2 | Lat_3 | Lat_4 | Circ_0 | Circ_1 | Circ_2 | Circ_3 | Circ_4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IID R10 vs NIID R50** | 0.8292 | 0.0050 | 0.3488 | 0.4023 | 0.3427 | 0.2397 | 0.4893 | 0.0000 | 0.0248 | 0.0000 | 0.0000 | 0.0000 |
| **NIID R10 vs NIID R50** | 0.6286 | 0.2000 | 0.1503 | 0.5005 | 0.2483 | 0.3015 | 0.3572 | 0.0000 | 0.0001 | 1.0000 | 0.0000 | 0.0000 |

### Inter-Client Similarity (NIID R50)
Compares the similarity between local models of different clients at the final round (Round 50).

| Comparison | Latent Total | Circuit Avg | Lat_0 | Lat_1 | Lat_2 | Lat_3 | Lat_4 | Circ_0 | Circ_1 | Circ_2 | Circ_3 | Circ_4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client 0 vs 1** | 1.0000 | 0.2703 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8747 | 0.0000 | 0.0000 | 0.0000 | 0.4770 |
| **Client 0 vs 2** | 1.0000 | 0.1700 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7815 | 0.0000 | 0.0000 | 0.0687 | 0.0000 |
| **Client 1 vs 2** | 1.0000 | 0.3770 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8826 | 1.0000 | 0.0023 | 0.0000 | 0.0000 |
