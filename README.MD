# RecipeRAG: A Knowledge Graph-Driven Approach to Personalized Recipe Retrieval and Generation

## Overview

This paper presents a benchmark designed to evaluate large language models (LLMs) based on their reasoning capabilities over ontology-based tasks. We focus on key Description Logic reasoning tasks such as subclass inference, class membership, and property assertions. The benchmark compares LLMs with traditional symbolic and neurosymbolic reasoners in terms of scalability, robustness, and explainability. Our findings reveal the strengths and limitations of LLMs in handling complex logical axioms and provide insights into their potential role as complements or alternatives to symbolic reasoners in knowledge-based systems.

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Instructions

1. Download the dataset (`dataFullLargerRegionAndCountryWithServingsBin.csv`) from [here](<https://drive.google.com/drive/folders/1_Tg9m87hRLKIMkC-xkkYNd6GpI-IX-mJ?usp=sharing>) and place it in the `/data` folder.

2. Folder `KG`:

- `knowledge_graph.py`: This module builds the knowledge graph using `dataFullLargerRegionAndCountryWithServingsBin.csv` as input. It outputs `data/triples_new_without_ct_ss.csv`, or you can access the precomputed file from [here](<https://drive.google.com/drive/folders/1_Tg9m87hRLKIMkC-xkkYNd6GpI-IX-mJ?usp=sharing>).

- `train_models.py`: This module trains knowledge graph embeddings. You can access the output from [here](<https://drive.google.com/drive/folders/1tvRAHzwq_qG77XBjyKNbpn6tff-gtNYy?usp=sharing>).

3. Folder `GroundTruth`:

- `build_ground_truth.ipynb`: This module builds the Test Set 1: standard queries without ingredient constraints. You can access the output from [here](<https://drive.google.com/drive/folders/1mEE8lOqc8q8SVvz652lmfH6qMm43uXRH>).

- `build_ground_truth_ingredient.ipynb`: This module builds the Test Set 2: ingredient-only queries. You can access the output from [here](<https://drive.google.com/drive/folders/1PbEjYdwD_211KxkpDUztVpcyrNav_bSi>).

4. Folder `Evaluation`:

- `kge_evaluation.py`: This module evaluates the knowledge graph embeddings by using Test Set 1.

- `kge_evaluation_ingredient.py`: This module evaluates the knowledge graph embeddings by using Test Set 2.

- `text_embedding_evaluation.py`: This module evaluates the text embeddings by using Test Set 1.

- `text_embedding_evaluation_ingredient.py`: This module evaluates the text embeddings by using Test Set 2.

5. Folder `Rag`:

- `experiments.ipynb`: This module employs a LLM-as-a-judge to evaluate the generation part of RecipeRAG. 
