# A Review of AI in Healthcare in the Era of Foundation Models
---
### Abstract
The proliferation of Electronic Health Records (EHR) presents a transformative opportunity for advancing healthcare delivery and research at the patient level. In the current era of foundation models, the landscape of clinical research is rapidly evolving, driven by innovations in Natural Language Processing (NLP) and deep learning techniques offering scalable and generalist AI capabilities. This paper provides a comprehensive overview of the state-of-the-art of EHR foundation models as well as frameworks derived from foundation models. The purpose of this work is to provide a scoping review of the latest developments in health research through the lens of emerging AI technologies. We further highlight challenges and opportunities for further development in this space.

---
# EHR Foundation Models

<details>
<summary> Encoder Foundation Models </summary>
  
### BERT-Based Representation Learning of Clinical and Scientific Data
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| BioBERT            | *Bioinformatics*                                                                          | [BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining](https://doi.org/10.1093/bioinformatics/btz682) |
| BioMedRoBERTa      | *Proceedings of ACL*                                                                      | [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/) |
| PubMedBERT         | *ACM Transactions on Computing for Healthcare*                                            | [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://dl.acm.org/doi/10.1145/3458754) |
| SciBERT            | *arXiv*                                                                                   | [A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676) |
| ClinicalBERT       | *arXiv*                                                                                   | [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342) |
| BioClinicalBERT    | *arXiv*                                                                                   | [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) |
| MedBERT (Version 1)| *APSIPA ASC*                                                                              | [MedBERT: A Pre-trained Language Model for Biomedical Named Entity Recognition](http://www.apsipa.org/proceedings/2022/APSIPA%202022/ThAM1-4/1570839765.pdf) |
| MedBERT (Version 2)| *NPJ Digital Medicine*                                                                    | [Med-BERT: Pretrained Contextualized Embeddings on Large-Scale Structured Electronic Health Records for Disease Prediction](https://www.nature.com/articles/s41746-021-00455-y) |
| RadBERT            | *Radiology: Artificial Intelligence*                                                      | [RadBERT: Adapting Transformer-based Language Models to Radiology](https://pubs.rsna.org/doi/full/10.1148/ryai.210258) |
| CEHR-BERT          | *Machine Learning for Health*                                                             | [CEHR-BERT: Incorporating Temporal Information from Structured EHR Data to Improve Prediction Tasks](https://proceedings.mlr.press/v158/) <br> *Note: Must look up paper on site to get PDF.* |

### BEHRT-Based temporal modeling
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| BEHRT              | *Scientific reports*                                                                       | [BEHRT: Transformer for Electronic Health Records](https://www.nature.com/articles/s41598-020-62922-y) |
| CORE-BEHRT          | *arXiv*                                                                                    | [CORE-BEHRT A Carefully Optimized and Rigorously Evaluated BEHRT](https://arxiv.org/html/2404.15201v2) |
| Multimodal BEHRT    | *medRxiv*                                                                                  | [Multimodal BEHRT: Transformers for Multimodal Electronic Health Records to predict breast cancer prognosis](https://www.medrxiv.org/content/10.1101/2024.09.18.24312984v1) |
| Hi-BEHRT           | *IEEE journal of biomedical and health informatics*                                        | [Hi-BEHRT: Hierarchical Transformer-Based Model for Accurate Prediction of Clinical Events Using Multimodal Longitudinal Electronic Health Records](https://pubmed.ncbi.nlm.nih.gov/36427286/)|
| Targeted BEHRT     | *IEEE Transactions on Neural Networks and Learning Systems*                                | [Targeted-BEHRT: Deep Learning for Observational Causal Inference on Longitudinal Electronic Health Records](https://pubmed.ncbi.nlm.nih.gov/35737602/) |
| ExBEHRT            | *International Workshop on Trustworthy Machine Learning for Healthcare*                   | [ExBEHRT: Extended Transformer for Electronic Health Records to Predict Disease Subtypes & Progressions](https://arxiv.org/abs/2303.12364) |
| MEME               | *arXiv*                                                                                    | [Multimodal Clinical Pseudo-notes for Emergency Department Prediction Tasks using Multiple Embedding Model for EHR (MEME)](https://arxiv.org/html/2402.00160v1)

### Other Transformer-Based Models
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| MiME               | *Advances in neural information processing systems*                                        | [MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare](https://papers.nips.cc/paper_files/paper/2018/hash/934b535800b1cba8f96a5d72f72f1611-Abstract.html)| 
| BioMegatron        | *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* | [BioMegatron: Larger Biomedical Domain Language Model](https://aclanthology.org/2020.emnlp-main.379/)
| GatorTron          | *arXiv*                                                                                    | [GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records](https://arxiv.org/abs/2203.03540) |

</details>

<details>
<summary> EHR Autoregressive Foundation Models </summary>

### Early Autoregressive Models
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| Doctor AI          | *Machine learning for healthcare conference*                                               | [Doctor AI: Predicting Clinical Events via Recurrent Neural Networks](https://pubmed.ncbi.nlm.nih.gov/28286600/) |

### GPT Based Models
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| CEHR-GPT           | *arXiv*                                                                                    | [CEHR-GPT: Generating Electronic Health Records with Chronological Patient Timelines](https://arxiv.org/abs/2402.04400) |
| Foresight          | *Lancet Digit. Health 6, e281–e290*                                                        | [Foresight—a generative pretrained transformer for modelling of patient timelines using electronic health records: a retrospective modelling study](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00025-6/fulltext) |
| Event Stream GPT   | *NIPS '23: Proceedings of the 37th International Conference on Neural Information Processing Systems* | [Event stream GPT: a data pre-processing and modeling library for generative, pre-trained transformers over continuous-time sequences of complex events](https://dl.acm.org/doi/10.5555/3666122.3667179) | 
| MedGPT             | *arXiv*                                                                                     | [MedGPT: Medical Concept Prediction from Clinical Narratives](https://arxiv.org/abs/2107.03134) |
| Hao et al. enhanced MedGPT| *Biomedical Informatics*                                                             | [A GPT-based EHR modeling system for unsupervised novel disease detection](https://www.sciencedirect.com/science/article/abs/pii/S1532046424001242)

### Modern Autoregressive Approaches
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| EHRMamba           | *arXiv*                                                                                    | [EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records](https://arxiv.org/abs/2405.14567) |
| ClinicalMamba      | *arXiv*                                                                                    | [ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes](https://arxiv.org/abs/2403.05795) |
| APRICOT-Mamba      | *arXiv*                                                                                    | [APRICOT-Mamba: Acuity Prediction in Intensive Care Unit (ICU): Development and Validation of a Stability, Transitions, and Life-Sustaining Therapies Prediction Model](https://arxiv.org/abs/2311.02026) |


</details>

<details>
<summary> EHR LLMs </summary>
  
### GPT Based LLMs
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| GatorTronGPT       | *NPJ Digital Medicine*                                                                     | [A study of generative large language model for medical research and healthcare](https://pubmed.ncbi.nlm.nih.gov/37973919/)
| ClinicalGPT        | *arXiv*                                                                                    | [ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation](https://arxiv.org/abs/2306.09968) |

### LLaMA Based LLMs 
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| ChatDoctor         | *Cureus*                                                                                   | [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://pmc.ncbi.nlm.nih.gov/articles/PMC10364849/) |
| MediTron-70B       | *arXiv*                                                                                    | [MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079) |
| PMC-LLaMA          | *arXiv*                                                                                    | [PMC-LLaMA: toward building open-source language models for medicine](https://arxiv.org/abs/2304.14454) |
| HuaTuo             | *arXiv*                                                                                    | [HuatuoGPT, Towards Taming Language Model to Be a Doctor](https://arxiv.org/abs/2305.15075) |

### Multimodal Medical LLMs
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| BiomedGPT          | *arXiv*                                                                                    | [BiomedGPT: A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks](https://arxiv.org/abs/2305.17100) |
| Med-Flamingo       | *arXiv*                                                                                    | [Med-Flamingo: a Multimodal Medical Few-shot Learner](https://arxiv.org/abs/2307.15189) | 
| LLaVA-MED          | *arXiv*                                                                                    | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) |
| MedPaLM 2          | *arXiv*                                                                                    | [Towards Expert-Level Medical Question Answering with Large Language Models](https://arxiv.org/abs/2305.09617) |

### Prompting Techniques in Medical LLMs
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| Dr. Knows          | *arXiv*                                                                                    | [Leveraging A Medical Knowledge Graph into Large Language Models for Diagnosis Prediction](https://arxiv.org/pdf/2308.14321v1) |
| ChatCAD            | *arXiv*                                                                                    | [ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models](https://arxiv.org/abs/2302.07257) |

### Other Medical LLMs
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| MedPaLM            | *Nature*                                                                                   | [Large language models encode clinical knowledge](https://www.nature.com/articles/s41586-023-06291-2) |
| NYUTron            | *Nature*                                                                                   | [Health system-scale language models are all-purpose prediction engines](https://pubmed.ncbi.nlm.nih.gov/37286606/) |
| MedCPT             | *Bioinformatics Oxford Academic*                                                           | [MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval](https://pubmed.ncbi.nlm.nih.gov/37930897/) |
| BioGPT             | *arXiv*                                                                                    | [BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining](https://arxiv.org/abs/2210.10341) |
| BioMistral         | *arXiv*                                                                                    | [BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains](https://arxiv.org/abs/2402.10373) |
| DRAGON             | *arXiv*                                                                                    | [Deep Bidirectional Language-Knowledge Graph Pretraining](https://arxiv.org/abs/2210.09338) |
| Med-Gemini         | *arXiv*                                                                                    | [Capabilities of Gemini Models in Medicine](https://arxiv.org/abs/2404.18416) |
| Clinical Camel     | *arXiv*                                                                                    | [Clinical Camel: An Open Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding](https://arxiv.org/abs/2305.12031) |
| Aloe               | *arXiv*                                                                                    | [Aloe: A Family of Fine-tuned Open Healthcare LLMs](https://arxiv.org/abs/2405.01886) |

</details>

# Clinical Foundation models

## Medical Imaging Foundation Modes

<details>
<summary> SAM Based Models </summary>
  
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| MedSAM             | *Nature Communications*                                                                    | [Segment anything in medical images](https://www.nature.com/articles/s41467-024-44824-z) |
| MedSAM 2           | *arXiv*                                                                                    | [Medical SAM 2: Segment medical images as video via Segment Anything Model 2](https://arxiv.org/abs/2408.00874) |
</details>

<details>
<summary> Other medical Imaging Foundation Models </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| VISION-MAE         | *arXiv*                                                                                    | [VISION-MAE: A Foundation Model for Medical Image Segmentation and Classification](https://arxiv.org/abs/2402.01034) |
| VISTA3D            | *arXiv*                                                                                    | [VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography](https://arxiv.org/abs/2406.05285) |
| GigaPath           | *Nature*                                                                                   | [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) |
| EVA-X              | *arXiv*                                                                                    | [EVA-X: A Foundation Model for General Chest X-ray Analysis with Self-supervised Learning](https://arxiv.org/abs/2405.05237) |
| Med-gemini         | *arXiv*                                                                                    | [Advancing Multimodal Medical Capabilities of Gemini](https://arxiv.org/abs/2405.03162) |
| PLIP               | *Nature Medical*                                                         | [A visual-language foundation model for pathology image analysis using medical Twitter](https://pubmed.ncbi.nlm.nih.gov/37592105/) |
| Med3D              | *arXiv*                                                                  | [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625) |
| BiomedCLIP         | *arXiv*                                                                  | [BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs](https://arxiv.org/abs/2303.00915) |
</details>

<details>
<summary> Frameworks </summary>
  
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| FairMedFM          | *arXiv*                                                                                    | [FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models](https://arxiv.org/abs/2407.00983) |
| SLIViT             | *Nature Biomedical Engineering*                                                            | [Accurate prediction of disease-https://www.biorxiv.org/content/10.1101/2023.01.11.523679v4risk factors from volumetric medical scans by a deep vision model pre-trained with 2D scans](https://ouci.dntb.gov.ua/en/works/7BYkgNZ9/) |
  
</details>

## Genetic & Genomics Foundation Modes

<details>
<summary> Genetic Foundation Models </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| Nucleotide Transformer | *bioRxiv*                                                                              | [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v4) |
| scBERT             | *Nature Machine Intelligence*                                                              | [scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data](https://www.nature.com/articles/s42256-022-00534-z) |
| scGPT              | *Nature Methods*                                                                           | [scGPT: toward building a foundation model for single-cell multi-omics using generative AI](https://www.nature.com/articles/s41592-024-02201-0) |
| SC-MAMBA2          | *bioRxiv*                                                                                  |[SC-MAMBA2: Leveraging State-Space Models for Efficient Single-Cell Ultra-Long Transcriptome Modeling](https://www.biorxiv.org/content/10.1101/2024.09.30.615775v1) |
</details>

<details>
<summary> Long Sequence Modeling </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| HyenaDNA           | *Advances in Neural Information Processing Systems*                                        | [HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution](https://dl.acm.org/doi/10.5555/3666122.3667994) | 
| GenaLM             | *bioRxiv*                                                                                  | [GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1) |
| Evo                | *bioRxiv*                                                                                  | [Sequence modeling and design from molecular to genome scale with Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1) |
</details>

<details>
<summary> Special Epigenetic Foundation models </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| CpGPT              | *bioRxiv*                                                                                  | [CpGPT: a Foundation Model for DNA Methylation](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1) |
| Orthrus            | *bioRxiv*                                                                                  | [Orthrus: Towards Evolutionary and Functional RNA Foundation Models](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v1.full)|
| Enformer           | *Nature Methods*                                                                           | [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x) |
</details>

## Physiological Signals and Waveforms Foundation Modes

<details>
<summary> ECG Foundation Models </summary>
  
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| ECG-FM             | *arXiv*                                                                                    | [ECG-FM: An Open Electrocardiogram Foundation Model](https://arxiv.org/abs/2408.05178) |
| HeAR               | *arXiv*                                                                                    | [HeAR -- Health Acoustic Representations](https://arxiv.org/abs/2403.02522) |
| PaPaGei            | *arXiv*                                                                                    | [PaPaGei: Open Foundation Models for Optical Physiological Signals](https://arxiv.org/abs/2410.20542) |
| SiamQuality        | *arXiv*                                                                                    | [SiamQuality: A ConvNet-Based Foundation Model for Imperfect Physiological Signals](https://arxiv.org/abs/2404.17667) |
| HeartBEiT          | *npj Digital Medicine*                                                                     | [A foundational vision transformer improves diagnostic performance for electrocardiograms](https://www.nature.com/articles/s41746-023-00840-9) | 
</details>

<details>
<summary> EEG Foundation models </summary>
  
| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| EEGFormer          | *arXiv*                                                                                    | [EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model](https://arxiv.org/abs/2401.10278) |
| The Generative Foundation Model for Five-Class Sleep Staging | *arXiv*                                          | [A generative foundation model for five-class sleep staging with arbitrary sensor input](https://arxiv.org/abs/2408.15253) |
| NeuroLM            | *arXiv*                                                                                    | [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/abs/2409.00101) |
| Nested Deep Learning Models | *arXiv*                                                                           | [Nested Deep Learning Model Towards A Foundation Model for Brain Signal Data](https://arxiv.org/abs/2410.03191) |
| EEGPT              | *arXiv*                                                                                    | [EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training](https://arxiv.org/abs/2410.19779) |
| BrainWave          | *arXiv*                                                                                    | [BrainWave: A Brain Signal Foundation Model for Clinical Applications](https://arxiv.org/abs/2402.10251) |

</details>

<details>
<summary> Multimodal Foundation models </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| Cross-Modal Representations | *arXiv*                                                                           | [Promoting cross-modal representations to improve multimodal foundation models for physiological signals](https://arxiv.org/abs/2410.16424) |
| Large-Scale Training for Wearable Biosignals | *arXiv*                                                          | [Large-scale Training of Foundation Models for Wearable Biosignals](https://arxiv.org/abs/2312.05409) |
| Foundation Models using Biosignals from Digital Stethoscopes | *npj Cardiovascular Health*                      | [Foundation models for cardiovascular disease detection via biosignals from digital stethoscopes](https://www.nature.com/articles/s44325-024-00027-5) |
| Universal ECG Foundation Models                    | *arXiv*                                                    | [An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains](https://arxiv.org/html/2410.04133) | 

</details>

<details>
<summary> Other Foundation models </summary>

| Model              | Source                                                                                     | Link |
|--------------------|--------------------------------------------------------------------------------------------|------|
| GluFormer          | *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*        | [Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification](https://arxiv.org/abs/2209.04526) |
</details>

### Contact

- Simon Lee (simonlee711@g.ucla.edu)
- Ava Gonick


  


