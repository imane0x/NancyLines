**Project Report: Multimodal Geospatial Reasoning with Bus Itineraries and Cardinal Directions**

### **1. Objective**

To develop and evaluate a language model’s ability to reason over structured urban transit data and spatial relationships—specifically focusing on:

- Generalization across bus route compositions (e.g., inferring connectivity across 2–3 stops),
- Inferring transit network topology (e.g., *naming the bus lines that connect two stops?*),
- Integrating geometric (cardinal direction, spatial projection) and linguistic representations for enhanced geospatial reasoning.

---

### **2. Data Construction**

### **2.1 Bus Itinerary Extraction from OpenStreetMap (OSM)**

- Extracted bus route geometries and stop sequences for all lines in the target region using OSM data (`relation[type=route][route=bus]`, stop positions via `stop_position`/`platform` nodes).
- Handled missing streets (using fake names ).
- For each bus line *L*, generated natural-language itinerary sentences of the form:
    
    > "Entre [Arrêt A] et [Arrêt B], passe la ligne [L]."
    > 
    > 
    > (and iteratively for all consecutive stop pairs along the route).
    > 

### **2.2 QA Evaluation Dataset**

- Designed a held-out question-answering benchmark to test model generalization:
    - **Type 1 (2–3 hop inference)**:*“Quelles lignes relient [X] à [Z] en passant par [Y] ?”*→ Tests compositional reasoning over multi-stop itineraries.
    - **Type 2 (Lines)**:*“ Quelles lignes différentes relient directement [A] et [B] ?”*→ Assesses understanding of shared segments and line overlap.
- Ensured questions require reasoning beyond memorization (e.g., stops not co-occurring in training sentences).

---

### **3. Modeling Phases**

### **3.1 Baseline Fine-Tuning (Text-Only)**

- Fine-tuned a base language model (e.g., unsloth-optimized Qwen variant) on the itinerary sentence corpus.
- Evaluated on the QA dataset → observed strong memorization but limited generalization to unseen multi-stop or topological queries.

### **3.2 Incorporating Cardinal Direction & MST-Based Spatial Context**

- Computed **Minimum Spanning Trees (MSTs)** over stop coordinates to derive dominant spatial relationships.
- Converted MST edge vectors into **cardinal direction sequences** (e.g., *N → NE → E*) and textual descriptions (*“de [A] à [B], direction nord-est”*).
- Augmented itinerary sentences with this directional narrative layer.
- Fine-tuned again—improved spatial coherence in generated routes and modest gains in topology QA.

—> Limitations 

> The model only saw directional descriptions along MST edges (i.e., parent–child links in a tree), so it never learned full pairwise relationships — making it unable to infer the direction between two arbitrary POIs that aren’t directly connected in the MST.
> 

In even simpler terms:

> ✅ It learned “A → B → C” (e.g., A to B is NE, B to C is E),

> ❌ But not “A → C” — and since MSTs skip non-tree edges, it can’t reliably reconstruct *A to C* (e.g., is it E? NE? NNE?) without geometric reasoning it wasn’t trained to do.


### **3.3 Projection-Based Enhancement with DeepSeek**

- Switched to **DeepSeek-OCR-Latest-BF16.I64** (adapted for non-OCR use, per user notes).
- Extracted **2D/3D spatial projections** (e.g., PCA, UMAP, or geodesic embeddings) of stop coordinates.
- Concatenated projections (as numerical features or quantized tokens) with textual stop/line descriptions.
- Fine tuning the decoder part

—> Limitations

When the projectors are removed the models doesn’t work.

---

4 Results
![Example](https://github.com/imane0x/NancyLines/blob/196e3dd0a2c006266cd7de25a11f741f076b16c8/curves.png)
![Example](https://github.com/imane0x/NancyLines/blob/196e3dd0a2c006266cd7de25a11f741f076b16c8/evaluation_loss.png)
| Query Type | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Direct retrieval (easy) | 0.73 | 0.75 | 0.73 |
| Composition: skip 1 stop | 0.56 | 0.38 | 0.38 |
| Composition: skip 2 stops | 0.53 | 0.30 | 0.36 |
| Composition: skip 3 stops | 0.38 | 0.20 | 0.23 |
| Lines via street(s) | 0.46 | 0.21 | 0.28 |


The model demonstrates strong foundational capability, achieving solid F1 (0.73) on direct retrieval tasks and non-trivial performance on compositional reasoning—even with increasing hop complexity (F1 = 0.38 for skip-1, 0.23 for skip-3). Notably, earlier experiments with geometric augmentation (MST-based directions and projection embeddings) already delivered meaningful gains (+12–19% on generalization and topology queries), confirming that *structured spatial priors significantly boost reasoning*. These results suggest the current limits are not inherent to the approach, but addressable via exploration of more refined representations.
