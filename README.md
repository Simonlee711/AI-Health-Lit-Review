# A Review of AI in Healthcare in the Era of Foundation Models
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
</details>

# Clinical Foundation models

## Medical Imaging Foundation Modes

<details>
<summary> SAM Based Models </summary>
[fill in]
</details>

<details>
<summary> Other medical Imaging Foundation Models </summary>
[fill in]
</details>

<details>
<summary> Frameworks </summary>
[fill in]
</details>

## Genetic & Genomics Foundation Modes

<details>
<summary> Genetic Foundation Models </summary>
[fill in]
</details>

<details>
<summary> Long Sequence Modeling </summary>
[fill in]
</details>

<details>
<summary> Special Eigenetic Foundation models </summary>
[fill in]
</details>

## Physiological Signals and Waveforms Foundation Modes

<details>
<summary> ECG Foundation Models </summary>
[fill in]
</details>

<details>
<summary> EEG Foundation models </summary>
[fill in]
</details>

<details>
<summary> Multimodal Foundation models </summary>
[fill in]
</details>

<details>
<summary> Other Foundation models </summary>
[fill in]
</details>

### Contact

- Simon Lee (simonlee711@g.ucla.edu)
- Ava Gonick


  


