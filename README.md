<h1 align="center">
🧬📝 Awesome Biomolecule-Language Cross Modeling
</h1>
<div align="center">

[![](https://img.shields.io/badge/paper-arxiv2403.01528-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2403.01528)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling?color=yellow&labelColor=555555)  ![Forks](https://img.shields.io/github/forks/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling?color=blue&label=Fork&labelColor=555555)
</div>

The repository for [Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey](https://arxiv.org/abs/2403.01528), including related models, datasets/benchmarks, and other resource links.

🔥 **We will keep this repository updated**.

🌟 **If you have a paper or resource you'd like to add, feel free to submit a pull request, open an issue, or email the author at <qizhipei@ruc.edu.cn>.**

<p align="center">
  <img src="figs/model_evolution.png" width="960">
</p>


## Table of Content
- [Models](#models)
  - [Biotext](#biotext)
  - [Text + Molecule](#text--molecule)
  - [Text + Protein](#text--protein)
  - [More Modalities](#more-modalities)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Related Resources](#related-resources)
  - [Related Surveys & Evaluations](#related-surveys--evaluations) 
  - [Related Repositories](#related-repositories)
- [Acknowledgements](#acknowledgements)
---

## Models
### Biotext
* **BioBERT: a pre-trained biomedical language representation model for biomedical text mining**
  
  [![](https://img.shields.io/badge/Bioinformatics_2019-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
  [![Stars](https://img.shields.io/github/stars/dmis-lab/biobert?color=yellow&style=social)](https://github.com/dmis-lab/biobert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/dmis-lab)

* **SciBERT: A Pretrained Language Model for Scientific Text**
  
  [![](https://img.shields.io/badge/EMNLP_IJCNLP_2019-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/D19-1371.pdf)
  [![Stars](https://img.shields.io/github/stars/allenai/scibert?color=yellow&style=social)](https://github.com/allenai/scibert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/allenai/scibert)

* **(BlueBERT) Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2019-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/W19-5006.pdf)
  [![Stars](https://img.shields.io/github/stars/ncbi-nlp/bluebert?color=yellow&style=social)](https://github.com/ncbi-nlp/bluebert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/bionlp)

* **Bio-Megatron: Larger Biomedical Domain Language Model**
  
  [![](https://img.shields.io/badge/EMNLP_2020-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2020.emnlp-main.379.pdf)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/NeMo?color=yellow&style=social)](https://github.com/NVIDIA/NeMo)

* **ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission**
  
  [![](https://img.shields.io/badge/BioNLP@CHIL_2020-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/1904.05342.pdf)
  [![Stars](https://img.shields.io/github/stars/kexinhuang12345/clinicalBERT?color=yellow&style=social)](https://github.com/kexinhuang12345/clinicalBERT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/lindvalllab/clinicalXLNet)

* **BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2021.bionlp-1.24.pdf)
  [![Stars](https://img.shields.io/github/stars/salrowili/BioM-Transformers?color=yellow&style=social)](https://github.com/salrowili/BioM-Transformers)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/sultan)

* **(PubMedBERT) Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing**
  
  [![](https://img.shields.io/badge/HEALTH_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://dl.acm.org/doi/10.1145/3458754)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

* **SciFive: a text-to-text transformer model for biomedical literature**
  
  [![](https://img.shields.io/badge/Arxiv_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2106.03598.pdf)
  [![Stars](https://img.shields.io/github/stars/justinphan3110/SciFive?color=yellow&style=social)](https://github.com/justinphan3110/SciFive)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/razent)

* **(DRAGON) Deep Bidirectional Language-Knowledge Graph Pretraining**
  
  [![](https://img.shields.io/badge/NeurIPS_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2210.09338.pdf)
  [![Stars](https://img.shields.io/github/stars/michiyasunaga/dragon?color=yellow&style=social)](https://github.com/michiyasunaga/dragon)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/michiyasunaga/dragon)

* **LinkBERT: Pretraining Language Models with Document Links**
  
  [![](https://img.shields.io/badge/ACL_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.acl-long.551.pdf)
  [![Stars](https://img.shields.io/github/stars/michiyasunaga/LinkBERT?color=yellow&style=social)](https://github.com/michiyasunaga/LinkBERT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/michiyasunaga)

* **BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.bionlp-1.9.pdf)
  [![Stars](https://img.shields.io/github/stars/GanjinZero/BioBART?color=yellow&style=social)](https://github.com/GanjinZero/BioBART)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GanjinZero)

* **BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining**
  
  [![](https://img.shields.io/badge/Bioinformatics_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1093/bib/bbac409)
  [![Stars](https://img.shields.io/github/stars/microsoft/BioGPT?color=yellow&style=social)](https://github.com/microsoft/BioGPT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/microsoft/biogpt)

* **GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2203.03540.pdf)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM?color=yellow&style=social)](https://github.com/NVIDIA/Megatron-LM)

* **Large language models encode clinical knowledge**
  
  [![](https://img.shields.io/badge/Nature_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41586-023-06291-2.pdf)

* **(ScholarBERT) The Diminishing Returns of Masked Language Models to Science**
  
  [![](https://img.shields.io/badge/ACL_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.findings-acl.82.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/globuslabs/ScholarBERT)

* **PMC-LLaMA: Further Finetuning LLaMA on Medical Papers**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.14454.pdf)
  [![Stars](https://img.shields.io/github/stars/chaoyi-wu/PMC-LLaMA?color=yellow&style=social)](https://github.com/chaoyi-wu/PMC-LLaMA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/chaoyi-wu/PMC-LLaMA)


* **BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.09442v2.pdf)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)

* **(GatortronGPT) A study of generative large language model for medical research and healthcare**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.13523.pdf)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM?color=yellow&style=social)](https://github.com/NVIDIA/Megatron-LM)

* **Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.12031.pdf)
  [![Stars](https://img.shields.io/github/stars/bowang-lab/clinical-camel?color=yellow&style=social)](https://github.com/bowang-lab/clinical-camel)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/wanglab/ClinicalCamel-70B)

* **MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.16079.pdf)
  [![Stars](https://img.shields.io/github/stars/epfLLM/meditron?color=yellow&style=social)](https://github.com/epfLLM/meditron)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/epfl-llm/meditron-70b)

* **BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-inspired Materials**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.08788.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/lamm-mit/BioinspiredLLM)

* **ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.09968.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/medicalai/ClinicalGPT-base-zh)

* **MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.08247.pdf)
  [![Stars](https://img.shields.io/github/stars/kbressem/medAlpaca?color=yellow&style=social)](https://github.com/kbressem/medAlpaca)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/medalpaca)

* **SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.07950.pdf)
  [![Stars](https://img.shields.io/github/stars/THUDM/SciGLM?color=yellow&style=social)](https://github.com/THUDM/SciGLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zd21/SciGLM-6B)

* **BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.18421v1.pdf)
  [![Stars](https://img.shields.io/github/stars/stanford-crfm/BioMedLM?color=yellow&style=social)](https://github.com/stanford-crfm/BioMedLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/stanford-crfm/BioMedLM)

* **BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.10373.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/BioMistral)

### Text + Molecule
* **Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries**
  
  [![](https://img.shields.io/badge/EMNLP_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2021.emnlp-main.47.pdf)
  [![Stars](https://img.shields.io/github/stars/cnedwards/text2mol?color=yellow&style=social)](https://github.com/cnedwards/text2mol)
  

* **(MolT5) Translation between Molecules and Natural Language**  
  
  [![](https://img.shields.io/badge/EMNLP_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.emnlp-main.26.pdf)
  [![Stars](https://img.shields.io/github/stars/blender-nlp/MolT5?color=yellow&style=social)](https://github.com/blender-nlp/MolT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/laituan245)

* **(KV-PLM) A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals**
  
  [![](https://img.shields.io/badge/Nature_Communications_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41467-022-28494-3.pdf)
  [![Stars](https://img.shields.io/github/stars/thunlp/KV-PLM?color=yellow&style=social)](https://github.com/thunlp/KV-PLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX)

* **(MoMu) A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2209.05481.pdf)
  [![Stars](https://img.shields.io/github/stars/BingSu12/MoMu?color=yellow&style=social)](https://github.com/BingSu12/MoMu)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/BingSu12/MoMu)

* **(Text+Chem T5) Unifying Molecular and Textual Representations via Multi-task Language Modelling**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/christofidellis23a/christofidellis23a.pdf)
  [![Stars](https://img.shields.io/github/stars/GT4SD/multitask_text_and_chemistry_t5?color=yellow&style=social)](https://github.com/GT4SD/multitask_text_and_chemistry_t5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GT4SD)

* **(CLAMP) Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/seidl23a/seidl23a.pdf)
  [![Stars](https://img.shields.io/github/stars/ml-jku/clamp?color=yellow&style=social)](https://github.com/ml-jku/clamp)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/ml-jku/clamp)

* **GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning**
  
  [![](https://img.shields.io/badge/NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=Tt6DrRCgJV)
  [![Stars](https://img.shields.io/github/stars/zhao-ht/GIMLET?color=yellow&style=social)](https://github.com/zhao-ht/GIMLET)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/haitengzhao/gimlet)

* **(HI-Mol) Data-Efficient Molecular Generation with Hierarchical Textual Inversion**
  
  [![](https://img.shields.io/badge/AI4D3@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=wwotGBxtC3)

* **MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction**
  
  [![](https://img.shields.io/badge/AI4D3@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ai4d3.github.io/papers/34.pdf)

* **(ChemLLMBench) What indeed can GPT models do in chemistry? A comprehensive benchmark on eight tasks**
  
  [![](https://img.shields.io/badge/Datasets&Benchmarks@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=1ngbR3SZHW)
  [![Stars](https://img.shields.io/github/stars/ChemFoundationModels/ChemLLMBench?color=yellow&style=social)](https://github.com/ChemFoundationModels/ChemLLMBench)

* **MolXPT: Wrapping Molecules with Text for Generative Pre-training**
  
  [![](https://img.shields.io/badge/ACL_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.acl-short.138.pdf)

* **(TextReact) Predictive Chemistry Augmented with Text Retrieval**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.784.pdf)
  [![Stars](https://img.shields.io/github/stars/thomas0809/textreact?color=yellow&style=social)](https://github.com/thomas0809/textreact)

* **MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.966v2.pdf)
  [![Stars](https://img.shields.io/github/stars/eltociear/MolCA?color=yellow&style=social)](https://github.com/eltociear/MolCA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://ufile.io/6vffm5bg)

* **ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.findings-emnlp.366.pdf)
  [![Stars](https://img.shields.io/github/stars/syr-cn/ReLM?color=yellow&style=social)](https://github.com/syr-cn/ReLM)

* **(MoleculeSTM) Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing**
  
  [![](https://img.shields.io/badge/Nature_Machine_Intelligence_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s42256-023-00759-6.pdf)
  [![Stars](https://img.shields.io/github/stars/chao1224/MoleculeSTM?color=yellow&style=social)](https://github.com/chao1224/MoleculeSTM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/chao1224/MoleculeSTM)

* **(AMAN) Adversarial Modality Alignment Network for Cross-Modal Molecule Retrieval**
  
  [![](https://img.shields.io/badge/IEEE_TAI_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/document/10063974)
  [![Stars](https://img.shields.io/github/stars/NicoleBonnie/AMAN?color=yellow&style=social)](https://github.com/NicoleBonnie/AMAN)

* **MolLM: A Unified Language Model for Integrating Biomedical Text with 2D and 3D Molecular Representations**
  
  [![](https://img.shields.io/badge/BioRxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2023.11.25.568656v2.full.pdf)
  [![Stars](https://img.shields.io/github/stars/gersteinlab/MolLM?color=yellow&style=social)](https://github.com/gersteinlab/MolLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/17XhqdsDOxiT8PEDLHdsLPKf62PXPmbms)

* **(MolReGPT) Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.06615.pdf)
  [![Stars](https://img.shields.io/github/stars/phenixace/MolReGPT?color=yellow&style=social)](https://github.com/phenixace/MolReGPT)

* **(CaR) Can Large Language Models Empower Molecular Property Prediction?**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.07443.pdf)
  [![Stars](https://img.shields.io/github/stars/ChnQ/LLM4Mol?color=yellow&style=social)](https://github.com/ChnQ/LLM4Mol)

* **MolFM: A Multimodal Molecular Foundation Model**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.09484.pdf)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://pan.baidu.com/share/init?surl=iAMBkuoZnNAylhopP5OgEg\&pwd=7a6b)

* **(ChatMol) Interactive Molecular Discovery with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.11976.pdf)
  [![Stars](https://img.shields.io/github/stars/Ellenzzn/ChatMol?color=yellow&style=social)](https://github.com/Ellenzzn/ChatMol)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/1I-LcBE0emj8p1W6WFCbEYajTADGN1RuC)

* **InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.16208.pdf)
  [![Stars](https://img.shields.io/github/stars/IDEA-XL/InstructMol?color=yellow&style=social)](https://github.com/IDEA-XL/InstructMol)

* **ChemCrow: Augmenting large-language models with chemistry tools**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.05376.pdf)
  [![Stars](https://img.shields.io/github/stars/ur-whitelab/chemcrow-public?color=yellow&style=social)](https://github.com/ur-whitelab/chemcrow-public)

* **GPT-MolBERTa: GPT Molecular Features Language Model for molecular property prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2310.03030.pdf)
  [![Stars](https://img.shields.io/github/stars/Suryanarayanan-Balaji/GPT-MolBERTa?color=yellow&style=social)](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa)

* **nach0: Multimodal Natural and Chemical Languages Foundation Model**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.12410.pdf)

* **DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.03907.pdf)
  [![Stars](https://img.shields.io/github/stars/UCSD-AI4H/drugchat?color=yellow&style=social)](https://github.com/UCSD-AI4H/drugchat)

* **(Ada/Aug-T5) From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.05203.pdf)
  [![Stars](https://img.shields.io/github/stars/SCIR-HI/ArtificiallyR2R?color=yellow&style=social)](https://github.com/SCIR-HI/ArtificiallyR2R)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/SCIR-HI)

* **MolTailor: Tailoring Chemical Molecular Representation to Specific Tasks via Text Prompts**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.11403.pdf)
  [![Stars](https://img.shields.io/github/stars/SCIR-HI/MolTailor?color=yellow&style=social)](https://github.com/SCIR-HI/MolTailor)

* **(TGM-DLM) Text-Guided Molecule Generation with Diffusion Language Model**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.13040.pdf)
  [![Stars](https://img.shields.io/github/stars/Deno-V/tgm-dlm?color=yellow&style=social)](https://github.com/Deno-V/tgm-dlm)

* **GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text**
  
  [![](https://img.shields.io/badge/Computers_in_Biology_and_Medicine_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1016/j.compbiomed.2024.108073)
  [![Stars](https://img.shields.io/github/stars/ai-hpc-research-team/git-mol?color=yellow&style=social)](https://github.com/ai-hpc-research-team/git-mol)

* **PolyNC: a natural and chemical language model for the prediction of unified polymer properties**
  
  [![](https://img.shields.io/badge/Chemical_Science_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://pubs.rsc.org/en/content/articlepdf/2024/sc/d3sc05079c)
  [![Stars](https://img.shields.io/github/stars/HKQiu/Unified_ML4Polymers?color=yellow&style=social)](https://github.com/HKQiu/Unified_ML4Polymers)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/hkqiu/PolyNC)

* **MolTC: Towards Molecular Relational Modeling In Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.03781.pdf)
  [![Stars](https://img.shields.io/github/stars/MangoKiller/MolTC?color=yellow&style=social)](https://github.com/MangoKiller/MolTC)

* **T-Rex: Text-assisted Retrosynthesis Prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.14637.pdf)
  [![Stars](https://img.shields.io/github/stars/lauyikfung/T-Rex?color=yellow&style=social)](https://github.com/lauyikfung/T-Rex)

* **LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09391.pdf)
  [![Stars](https://img.shields.io/github/stars/OSU-NLP-Group/LLM4Chem?color=yellow&style=social)](https://github.com/OSU-NLP-Group/LLM4Chem)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/osunlp/LLM4Chem)

* **(Drug-to-indication) Emerging Opportunities of Using Large Language Models for Translation Between Drug Molecules and Indications**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09588.pdf)
  [![Stars](https://img.shields.io/github/stars/PittNAIL/drug-to-indication?color=yellow&style=social)](https://github.com/PittNAIL/drug-to-indication)

* **ChemDFM: Dialogue Foundation Model for Chemistry**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.14818.pdf)

* **DrugAssist: A Large Language Model for Molecule Optimization**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.10334.pdf)
  [![Stars](https://img.shields.io/github/stars/blazerye/DrugAssist?color=yellow&style=social)](https://github.com/blazerye/DrugAssist)

* **ChemLLM: A Chemical Large Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.06852.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)

* **(TEDMol) Text-guided Diffusion Model for 3D Molecule Generation**
  
  [![](https://img.shields.io/badge/OpenReview-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=FdUloEgBSE)

* **(3DToMolo) Sculpting Molecules in 3D: A Flexible Substructure Aware Framework for Text-Oriented Molecular Optimization**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.03425.pdf)

* **(ICMA) Large Language Models are In-Context Molecule Learners**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.04197.pdf)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/phenixace)

* **Benchmarking Large Language Models for Molecule Prediction Tasks**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.05075.pdf)
  [![Stars](https://img.shields.io/github/stars/zhiqiangzhongddu/LLMaMol?color=yellow&style=social)](https://github.com/zhiqiangzhongddu/LLMaMol)

* **3M-Diffusion: Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.07179.pdf)
  [![Stars](https://img.shields.io/github/stars/huaishengzhu/3MDiffusion?color=yellow&style=social)](https://github.com/huaishengzhu/3MDiffusion)

* **(TSMMG) Instruction Multi-Constraint Molecular Generation Using a Teacher-Student Large Language Model**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.13244.pdf)
  [![Stars](https://img.shields.io/github/stars/HHW-zhou/TSMMG?color=yellow&style=social)](https://github.com/HHW-zhou/TSMMG)

* **(SLM4CRP) A Self-feedback Knowledge Elicitation Approach for Chemical Reaction Predictions**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2404.09606.pdf)
  [![Stars](https://img.shields.io/github/stars/AI-HPC-Research-Team/SLM4CRP?color=yellow&style=social)](https://github.com/AI-HPC-Research-Team/SLM4CRP)

* **Atomas: Hierarchical Alignment on Molecule-Text for Unified Molecule Understanding and Generation**
  
  [![](https://img.shields.io/badge/ICLR_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2404.16880.pdf)
  [![Stars](https://img.shields.io/github/stars/yikunpku/Atomas?color=yellow&style=social)](https://github.com/yikunpku/Atomas)

* **ReactXT: Understanding Molecular"Reaction-ship"via Reaction-Contextualized Molecule-Text Pretraining**
  
  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.14225)
  [![Stars](https://img.shields.io/github/stars/syr-cn/ReactXT?color=yellow&style=social)](https://github.com/syr-cn/ReactXT)

* **(ALMol) Aligned Language-Molecule Translation LLMs through Offline Preference Contrastive Optimisation**  

  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.langmol-1.3/)
  [![Stars](https://img.shields.io/github/stars/REAL-Lab-NU/Awesome-LLM-Centric-Molecular-Discovery?color=yellow&style=social)](https://github.com/REAL-Lab-NUgithub.com/REAL-Lab-NU/Awesome-LLM-Centric-Molecular-Discovery)

* **LDMol: Text-Conditioned Molecule Diffusion Model Leveraging Chemically Informative Latent Space**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.17829)
  [![Stars](https://img.shields.io/github/stars/jinhojsk515/LDMol?color=yellow&style=social)](https://github.com/jinhojsk515/LDMol)

* **DrugLLM: Open Large Language Model for Few-shot Molecule Generation**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.06690)

* **(HI-Mol) Data-Efficient Molecular Generation with Hierarchical Textual Inversion**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2405.02845)
  [![Stars](https://img.shields.io/github/stars/Seojin-Kim/HI-Mol?color=yellow&style=social)](https://github.com/Seojin-Kim/HI-Mol)

* **(MV-Mol) Learning Multi-view Molecular Representations with Structured and Unstructured Knowledge**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.09841)
  [![Stars](https://img.shields.io/github/stars/icycookies/MV-Mol?color=yellow&style=social)](https://github.com/icycookies/MV-Mol)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/1cd1EZTuyNOCLRt2Vr0JgXh3kWm9f2qwU)

* **DRAK: Unlocking Molecular Insights with Domain-Specific Retrieval-Augmented Knowledge in LLMs**

  [![](https://img.shields.io/badge/ResearchGate_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.18535)

* **HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.14021)
  [![Stars](https://img.shields.io/github/stars/LFhase/HIGHT?color=yellow&style=social)](https://github.com/LFhase/HIGHT)

* **PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes**
  
  [![](https://img.shields.io/badge/EMNLP_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.13193v1)
  [![Stars](https://img.shields.io/github/stars/IDEA-XL/PRESTO?color=yellow&style=social)](https://github.com/IDEA-XL/PRESTO)

* **3D-MolT5: Towards Unified 3D Molecule-Text Modeling with 3D Molecular Tokenization**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.05797)
  [![Stars](https://img.shields.io/github/stars/QizhiPei/3D-MolT5?color=yellow&style=social)](https://github.com/QizhiPei/3D-MolT5)

* **MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.12950)
  [![Stars](https://img.shields.io/github/stars/NYUSHCS/MolecularGPT?color=yellow&style=social)](https://github.com/NYUSHCS/MolecularGPT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/YuyanLiu/MolecularGPT)

* **MolX: Enhancing Large Language Models for Molecular Learning with A Multi-Modal Extension**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2406.06777)

* **(AMOLE) Vision Language Model is NOT All You Need: Augmentation Strategies for Molecule Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2407.09043)
  [![Stars](https://img.shields.io/github/stars/Namkyeong/AMOLE?color=yellow&style=social)](https://github.com/Namkyeong/AMOLE)

* **(Chemma-RC) Text-Augmented Multimodal LLMs for Chemical Reaction Condition Recommendation**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2407.15141)

* **Chemical Language Models Have Problems with Chemistry: A Case Study on Molecule Captioning Task**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=JoO6mtCLHD)
  [![Stars](https://img.shields.io/github/stars/ChemistryLLMs/SMILES-probing?color=yellow&style=social)](https://github.com/ChemistryLLMs/SMILES-probing)

* **UniMoT: Unified Molecule-Text Language Model with Discrete Token Representation**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2408.00863)
  [![Stars](https://img.shields.io/github/stars/Uni-MoT/uni-mot.github.io?color=yellow&style=social)](https://github.com/Uni-MoT/uni-mot.github.io)  

* **(UTGDiff) Instruction-Based Molecular Graph Generation with Unified Text-Graph Diffusion Model**  

  [![](https://img.shields.io/badge/ECAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2408.09896) 
  [![Stars](https://img.shields.io/github/stars/ran1812/UTGDiff?color=yellow&style=social)](https://github.com/ran1812/UTGDiff) 
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/18EqQ7MDHesmtiMiZz2o09PyeSwyf0hXb)  

* **Mol2Lang-VLM: Vision- and Text-Guided Generative Pre-trained Language Models for Advancing Molecule Captioning through Multimodal Fusion**  

  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.langmol-1.12/)
  [![Stars](https://img.shields.io/github/stars/nhattruongpham/mol-lang-bridge?color=yellow&style=social)](https://github.com/nhattruongpham/mol-lang-bridge/tree/mol2lang/)

* **Lang2Mol-Diff: A Diffusion-Based Generative Model for Language-to-Molecule Translation Leveraging SELFIES Representation**  

  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.langmol-1.15/)
  [![Stars](https://img.shields.io/github/stars/nhattruongpham/mol-lang-bridge?color=yellow&style=social)](https://github.com/nhattruongpham/mol-lang-bridge/tree/lang2mol/)

* **Enhancing Cross Text-Molecule Learning by Self-Augmentation**  

  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.findings-acl.569/)

* **MTSwitch: A Web-based System for Translation between Molecules and Texts**  

  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.inlg-demos.2.pdf)
  [![Stars](https://img.shields.io/github/stars/hanninaa/MTSwitch?color=yellow&style=social)](https://github.com/hanninaa/MTSwitch)

* **SmileyLlama: Modifying Large Language Models for Directed Chemical Space Exploration**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2409.02231)

* **(MoleculeSTM) Geometry-text Multi-modal Foundation Model for Reactivity-oriented Molecule Editing**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=A9FlQMKxJ4)
  [![Stars](https://img.shields.io/github/stars/chao1224/MoleculeSTM?color=yellow&style=social)](https://github.com/chao1224/MoleculeSTM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/chao1224/MoleculeSTM/tree/main)

* **(MSR) Structural Reasoning Improves Molecular Understanding of LLM**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2410.05610)
  [![Stars](https://img.shields.io/github/stars/yunhuijang/MSR?color=yellow&style=social)](https://github.com/yunhuijang/MSR)

* **(TransDLM) Text-Guided Multi-Property Molecular Optimization with a Diffusion Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2410.13597)
  [![Stars](https://img.shields.io/github/stars/Cello2195/TransDLM?color=yellow&style=social)](https://github.com/Cello2195/TransDLM)

* **Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2410.03138)

* **(ChemLML) Chemical Language Model Linker: blending text and molecules with modular adapters**
  
  [![](https://img.shields.io/badge/JCIM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2410.20182)
  [![Stars](https://img.shields.io/github/stars/gitter-lab/ChemLML?color=yellow&style=social)](https://github.com/gitter-lab/ChemLML)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://doi.org/10.5281/zenodo.11661517)

* **Small Molecule Optimization with Large Language Models**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=nJCYKdRZXb)
  [![Stars](https://img.shields.io/github/stars/yerevann/chemlactica?color=yellow&style=social)](https://github.com/yerevann/chemlactica)

* **Question Rephrasing for Quantifying Uncertainty in Large Language Models: Applications in Molecular Chemistry Tasks**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://neurips.cc/media/neurips-2024/Slides/105558.pdf)

* **(LLaMo) Large Language Model-based Molecular Graph Assistant**  

  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.neurips.cc/paper_files/paper/2024/file/ee46288ab2aaf5c6e53aebebe719712c-Paper-Conference.pdf)
  [![Stars](https://img.shields.io/github/stars/mlvlab/LLaMo?color=yellow&style=social)](https://github.com/mlvlab/LLaMo)

* **(M³LLM) Exploring Hierarchical Molecular Graph Representation in Multimodal LLMs**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2411.04708)

* **MolReFlect: Towards In-Context Fine-grained Alignments between Molecules and Texts**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2411.14721)

* **(AMORE) Lost in Translation: Chemical Language Models and the Misunderstanding of Molecule Structures**
  
  [![](https://img.shields.io/badge/ENMLP_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2024.findings-emnlp.760/)
  [![Stars](https://img.shields.io/github/stars/ChemistryLLMs/AMORE?color=yellow&style=social)](https://github.com/ChemistryLLMs/AMORE)

* **GeomCLIP: Contrastive Geometry-Text Pre-training for Molecules**
  
  [![](https://img.shields.io/badge/BIBM_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/10822346)
  [![Stars](https://img.shields.io/github/stars/xiaocui3737/GeomCLIP?color=yellow&style=social)](https://github.com/xiaocui3737/GeomCLIP)

* **(CMTMR) Towards Cross-Modal Text-Molecule Retrieval with Better Modality Alignment**
  
  [![](https://img.shields.io/badge/BIBM_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/10821722)
  [![Stars](https://img.shields.io/github/stars/XMUDeepLIT/CMTMR?color=yellow&style=social)](https://github.com/XMUDeepLIT/CMTMR)

* **(ORMA) Exploring Optimal Transport-Based Multi-Grained Alignments for Text-Molecule Retrieval**
  
  [![](https://img.shields.io/badge/BIBM_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/10822800)
  [![Stars](https://img.shields.io/github/stars/XMUDeepLIT/Orma?color=yellow&style=social)](https://github.com/XMUDeepLIT/Orma)

* **PEIT: Property Enhanced Instruction Tuning for Multi-task Molecular Generation with LLMs**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2412.18084)
  [![Stars](https://img.shields.io/github/stars/chenlong164/PEIT?color=yellow&style=social)](https://github.com/chenlong164/PEIT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main)

* **(HME) Navigating Chemical-Linguistic Sharing Space with Heterogeneous Molecular Encoding**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2412.20888)
  [![Stars](https://img.shields.io/github/stars/Lyu6PosHao/HME?color=yellow&style=social)](https://github.com/Lyu6PosHao/HME)

* **(Llamole) Multimodal Large Language3D-MOLT5 Models for Inverse Molecular Design with Retrosynthetic Planning**
  
  [![](https://img.shields.io/badge/ICLR_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=rQ7fz9NO7f)
  [![Stars](https://img.shields.io/github/stars/liugangcode/Llamole?color=yellow&style=social)](https://github.com/liugangcode/Llamole)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/collections/liuganghuggingface/llamole-collection)

* **RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning**
  
  [![](https://img.shields.io/badge/ICLR_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=J6e4hurEKd)
  [![Stars](https://img.shields.io/github/stars/Kin-CL/RetroInText?color=yellow&style=social)](https://github.com/Kin-CL/RetroInText)

* **OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2501.15415)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OCSU?color=yellow&style=social)](https://github.com/PharMolix/OCSU)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/PharMolix/Mol-VL-7B)

* **Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks**  

  [![](https://img.shields.io/badge/TCSS_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.01074)
  [![Stars](https://img.shields.io/github/stars/1789336421/Omni-Mol?color=yellow&style=social)](https://github.com/1789336421/Omni-Mol)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/datasets/CodeMagic/Omni-Mol-Dataset)

* **Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.02810)

* **(SLM4Mol) A Quantitative Analysis of Knowledge-Learning Preferences in Large Language Models in Molecular Science**
  
  [![](https://img.shields.io/badge/NMI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s42256-024-00977-6)
  [![Stars](https://img.shields.io/github/stars/AI-HPC-Research-Team/SLM4Mol?color=yellow&style=social)](https://github.com/AI-HPC-Research-Team/SLM4Mol)

* **CLASS: Enhancing Cross-Modal Text-Molecule Retrieval Performance and Training Efficiency**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.11633)

* **Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.13449)
  [![Stars](https://img.shields.io/github/stars/DongkiKim95/Mol-LLaMA?color=yellow&style=social)](https://github.com/DongkiKim95/Mol-LLaMA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/DongkiKim/Mol-Llama-2-7b-chat)

* **ChatMol: A Versatile Molecule Designer Based on the Numerically Enhanced Large Language Model**
  
  [![](https://img.shields.io/badge/Bioinformatics_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.19794)
  [![Stars](https://img.shields.io/github/stars/ChatMol/ChatMol?color=yellow&style=social)](https://github.com/ChatMol/ChatMol)

* **MV-CLAM: Multi-View Molecular Interpretation with Cross-Modal Projection via Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2503.04780)
  [![Stars](https://img.shields.io/github/stars/sumin124/mv-clam?color=yellow&style=social)](https://github.com/sumin124/mv-clam)

* **GraphT5: Unified Molecular Graph-Language Modeling via Multi-Modal Cross-Token Attention**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2503.07655)

* **XMolCap: Advancing Molecular Captioning through Multimodal Fusion and Explainable Graph Neural Networks**
  
  [![](https://img.shields.io/badge/JBHI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/11012653)
  [![Stars](https://img.shields.io/github/stars/cbbl-skku-org/XMolCap?color=yellow&style=social)](https://github.com/cbbl-skku-org/XMolCap)

* **(ChatChemTS) Large language models open new way of AI-assisted molecule design for chemists**
  
  [![](https://img.shields.io/badge/JCIM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/11012653)
  [![Stars](https://img.shields.io/github/stars/molecule-generator-collection/ChatChemTS?color=yellow&style=social)](https://github.com/molecule-generator-collection/ChatChemTS)

* **(LLM-MPP) Effective and Explainable Molecular Property Prediction by Chain-of-Thought Enabled Large Language Models and Multi-Modal Molecular Information Fusion**
  
  [![](https://img.shields.io/badge/JCIM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00577)
  [![Stars](https://img.shields.io/github/stars/jinchang1223/LLM-MPP?color=yellow&style=social)](https://github.com/jinchang1223/LLM-MPP)

* **Graph2Token: Make LLMs Understand Molecule Graphs**  

  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1609/aaai.v39i20.35422)
  [![Stars](https://img.shields.io/github/stars/ZeLeBron/Graph2Token?color=yellow&style=social)](https://github.com/ZeLeBron/Graph2Token)  

* **ExDDI: Explaining Drug-Drug Interaction Predictions with Natural Language**  

  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/34709)
  [![Stars](https://img.shields.io/github/stars/ZhaoyueSun/ExDDI?color=yellow&style=social)](https://github.com/ZhaoyueSun/ExDDI)

* **ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area**  

  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/32020)
  [![Stars](https://img.shields.io/github/stars/lijunxian111/ChemVlm?color=yellow&style=social)](https://github.com/lijunxian111/ChemVlm)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/AI4Chem/ChemVLM-26B-1-2)

* **ChemDual: Enhancing Chemical Reaction and Retrosynthesis Prediction with Large Language Model and Dual-task Learning**
  
  [![](https://img.shields.io/badge/ICJAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.02639)
  [![Stars](https://img.shields.io/github/stars/JacklinGroup/ChemDual?color=yellow&style=social)](https://github.com/JacklinGroup/ChemDual)

* **GeLLM³O: Generalizing Large Language Models for Multi-property Molecule Optimization**  

  [![](https://img.shields.io/badge/ACL_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2025.acl-long.1225.pdf)
  [![Stars](https://img.shields.io/github/stars/ninglab/GeLLMO?color=yellow&style=social)](https://github.com/ninglab/GeLLMO)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/NingLab/GeLLMO-P6-Mistral)

* **Less for More: Enhanced Feedback-aligned Mixed LLMs for Molecule Caption Generation and Fine-Grained NLI Evaluation**  

  [![](https://img.shields.io/badge/ACL_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2025.acl-long.144/)

* **(GeLLMO-C) Large Language Models for Controllable Multi-property Multi-objective Molecule Optimization**  

  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.23987)
  [![Stars](https://img.shields.io/github/stars/ninglab/GeLLMO-C?color=yellow&style=social)](https://github.com/ninglab/GeLLMO-C)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/collections/NingLab/controllable-gellmo)

* **ReactGPT: Understanding of Chemical Reactions via In-Context Tuning**  

  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/31983)

* **nach0-pc: Multi-task Language Model with Molecular Point Cloud Encoder**  

  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1609/aaai.v39i23.34613)

* **mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.12565)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/collections/language-plus-molecules/mclm)

* **ChemMLLM: Chemical Multimodal Large Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.16326)
  [![Stars](https://img.shields.io/github/stars/bbsbz/ChemMLLM?color=yellow&style=social)](https://github.com/bbsbz/ChemMLLM)

* **(CLEANMOL) Improving Chemical Understanding of LLMs via SMILES Parsing**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.16340)

* **ModuLM: Enabling Modular and Multimodal Molecular Relational Learning with Large Language Models**
  
  [![](https://img.shields.io/badge/NeurIPS_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/html/2506.00880v1)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://anonymous.4open.science/r/ModuLM/README.md)

* **(ToDi) TextOmics-Guided Diffusion for Hit-like Molecular Generation**  

  [![](https://img.shields.io/badge/arXiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2507.09982)

* **ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2507.21990v2)

* **Dual Learning Between Molecules and Natural Language**
  
  [![](https://img.shields.io/badge/PAKDD_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://link.springer.com/chapter/10.1007/978-981-96-8173-0_31)

* **CROP: Integrating Topological and Spatial Structures via Cross-View Prefixes for Molecular LLMs**
  
  [![](https://img.shields.io/badge/MM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2508.06917)

* **Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2508.08401)

* **AttriLens-Mol: Attribute Guided Reinforcement Learning for Molecular Property Prediction with Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2508.04748)
  [![Stars](https://img.shields.io/github/stars/szu-tera/AttriLens-Mol?color=yellow&style=social)](https://github.com/szu-tera/AttriLens-Mol)

* **MolPrompt: improving multi-modal molecular pre-training with knowledge prompts**
  
  [![](https://img.shields.io/badge/Bioinformatics_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://academic.oup.com/bioinformatics/article/41/9/btaf466/8240326?login=false)
  [![Stars](https://img.shields.io/github/stars/catly/MolPrompt?color=yellow&style=social)](https://github.com/catly/MolPrompt)

* **(CAMT5) Training Text-to-Molecule Models with Context-Aware Tokenization**
  
  [![](https://img.shields.io/badge/ENMLP_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2509.04476)
  [![Stars](https://img.shields.io/github/stars/Songhyeontae/CAMT5?color=yellow&style=social)](https://github.com/Songhyeontae/CAMT5)

* **Enhancing Molecular Property Prediction with Knowledge from Large Language Models**

  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2509.20664)

* **(MolFinePrompt) Fine-grained multimodal molecular pretraining via prompt learning**
  
  [![](https://img.shields.io/badge/Knosys_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1016/j.knosys.2025.114381) 
  [![Stars](https://img.shields.io/github/stars/yzf-code/KnowMol?color=yellow&style=social)](https://github.com/catly/MolFinePrompt)

* **(MPPReasoner) Reasoning-Enhanced Large Language Models for Molecular Property Prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2510.10248)
  [![Stars](https://img.shields.io/github/stars/Jesse-zjx/MPPReasoner?color=yellow&style=social)](https://github.com/Jesse-zjx/MPPReasoner)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://anonymous.4open.science/r/MPPReasoner-12687/README.md)

* **(MECo) Coder as Editor: Code-Driven Interpretable Molecule Optimization**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2510.14455)

* **Chem-R: Learning to Reason as a Chemist**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2510.16880)
  [![Stars](https://img.shields.io/github/stars/davidweidawang/Chem-R?color=yellow&style=social)](https://github.com/davidweidawang/Chem-R)

* **KnowMol: Advancing Molecular Large Language Models with Multi-Level Chemical Knowledge**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2510.19484)
  [![Stars](https://img.shields.io/github/stars/yzf-code/KnowMol?color=yellow&style=social)](https://github.com/yzf-code/KnowMol)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://hf.co/datasets/yzf1102/KnowMol-100K)

* **(Mol-LLM) Incorporating Molecular Knowledge in Large Language Models via Multimodal Modeling**
  
  [![](https://img.shields.io/badge/TCSS_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/abstract/document/10838383)

* **DeepMolTex: Deep Alignment of Molecular Graphs with Large Language Models via Mixture of Modality Experts**
  
  [![](https://img.shields.io/badge/MM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://dl.acm.org/doi/abs/10.1145/3746027.3755875)

* **Mol-L2: Transferring text knowledge with frozen language models for molecular representation learning**
  
  [![](https://img.shields.io/badge/Neurocomputing_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015097)

* **(MolRAG) Unlocking the Power of LLMs for Molecular Property Prediction**  

  [![](https://img.shields.io/badge/ACL_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2025.acl-long.755/) 
  [![Stars](https://img.shields.io/github/stars/AcaciaSin/MolRAG?color=yellow&style=social)](https://github.com/AcaciaSin/MolRAG)  


### Text + Protein
* **OntoProtein: Protein Pretraining With Gene Ontology Embedding**
  
  [![](https://img.shields.io/badge/ICLR_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=yfe1VMYAXa4)
  [![Stars](https://img.shields.io/github/stars/zjunlp/OntoProtein?color=yellow&style=social)](https://github.com/zjunlp/OntoProtein)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp/OntoProtein)

* **ProTranslator: Zero-Shot Protein Function Prediction Using Textual Description**
  
  [![](https://img.shields.io/badge/RECOMB_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2204.10286)
  [![Stars](https://img.shields.io/github/stars/HanwenXuTHU/ProTranslator?color=yellow&style=social)](https://github.com/HanwenXuTHU/ProTranslator)

* **ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/xu23t/xu23t.pdf)
  [![Stars](https://img.shields.io/github/stars/DeepGraphLearning/ProtST?color=yellow&style=social)](https://github.com/DeepGraphLearning/ProtST)

* **(ProGen) Large language models generate functional protein sequences across diverse families**
  
  [![](https://img.shields.io/badge/Nature_Biotechnology_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41587-022-01618-2)
  [![Stars](https://img.shields.io/github/stars/salesforce/progen?color=yellow&style=social)](https://github.com/salesforce/progen)
  
* **InstructProtein: Aligning Human and Protein Language via Knowledge Instruction**
  
  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2310.03269.pdf)
  [![Stars](https://img.shields.io/github/stars/DeepGraphLearning/ProtST?color=yellow&style=social)](https://github.com/HICAI-ZJU/InstructProtein)

* **(ProteinDT) A Text-guided Protein Design Framework**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2302.04611.pdf)
  [![Stars](https://img.shields.io/github/stars/chao1224/ProteinDT?color=yellow&style=social)](https://github.com/chao1224/ProteinDT)

* **ProteinChat: Towards Achieving ChatGPT-Like Functionalities on Protein 3D Structures**
  
  [![](https://img.shields.io/badge/TechRxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23120606.v1)
  [![Stars](https://img.shields.io/github/stars/UCSD-AI4H/proteinchat?color=yellow&style=social)](https://github.com/UCSD-AI4H/proteinchat)

* **Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.14367.pdf)
  [![Stars](https://img.shields.io/github/stars/hadi-abdine/Prot2Text?color=yellow&style=social)](https://github.com/hadi-abdine/Prot2Text)
  
* **ProtChatGPT: Towards Understanding Proteins with Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09649.pdf)

* **ProtAgents: Protein discovery via large language model multi-agent collaborations combining physics and machine learning**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.04268.pdf)
  [![Stars](https://img.shields.io/github/stars/lamm-mit/ProtAgents?color=yellow&style=social)](https://github.com/lamm-mit/ProtAgents)

* **ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.16445v1.pdf)
  [![Stars](https://img.shields.io/github/stars/Lyu6PosHao/ProLLaMA?color=yellow&style=social)](https://github.com/Lyu6PosHao/ProLLaMA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GreatCaptainNemo/ProLLaMA)

* **ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training**
  
  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.07920.pdf)
  [![Stars](https://img.shields.io/github/stars/ProtLLM/ProtLLM?color=yellow&style=social)](https://github.com/ProtLLM/ProtLLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/datasets/ProtLLM/ProtLLM)

* **ProtT3: Protein-to-Text Generation for Text-based Protein Understanding**
  
  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.12564)
  [![Stars](https://img.shields.io/github/stars/acharkq/ProtT3?color=yellow&style=social)](https://github.com/acharkq/ProtT3)

* **ProteinCLIP: enhancing protein language models with natural language**
  
  [![](https://img.shields.io/badge/bioRxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2024.05.14.594226v1.full.pdf)
  [![Stars](https://img.shields.io/github/stars/wukevin/proteinclip?color=yellow&style=social)](https://github.com/wukevin/proteinclip)

* **ProLLM: Protein Chain-of-Thoughts Enhanced LLM for Protein-Protein Interaction Prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.06649)
  [![Stars](https://img.shields.io/github/stars/MingyuJ666/ProLLM?color=yellow&style=social)](https://github.com/MingyuJ666/ProLLM)

* **(PAAG) Functional Protein Design with Local Domain Alignment**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2404.16866)

* **(Pinal) Toward De Novo Protein Design from Natural Language**
  
  [![](https://img.shields.io/badge/bioRrxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2024.08.01.606258v1.full.pdf)

* **TourSynbio: A Multi-Modal Large Model and Agent Framework to Bridge Text and Protein Sequences for Protein Engineering**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.arxiv.org/pdf/2408.15299)
  [![Stars](https://img.shields.io/github/stars/tsynbio/TourSynbio?color=yellow&style=social)](https://github.com/tsynbio/TourSynbio)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/tsynbio/Toursynbio)

* **(SEPIT) Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding with LLMs**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2410.03553)
  [![Stars](https://img.shields.io/github/stars/U-rara/SEPIT?color=yellow&style=social)](https://github.com/U-rara/SEPIT)

* **(BioM3) Natural Language Prompts Guide the Design of Novel Functional Protein Sequences**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=L1MyyRCAjX)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/niksapraljak1/BioM3)

* **Language Models for Text-guided Protein Evolution**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=CNla8z0V2p)
  [![Stars](https://img.shields.io/github/stars/ZhanghanNi/LLM4ProteinEvolution?color=yellow&style=social)](https://github.com/ZhanghanNi/LLM4ProteinEvolution)

* **MMSite: A Multi-modal Framework for the Identification of Active Sites in Proteins**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://neurips.cc/virtual/2024/poster/94780)
  [![Stars](https://img.shields.io/github/stars/Gift-OYS/MMSite?color=yellow&style=social)](https://github.com/Gift-OYS/MMSite)

* **MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://neurips.cc/virtual/2024/poster/92987)
  [![Stars](https://img.shields.io/github/stars/PharMolix/MutaPLM?color=yellow&style=social)](https://github.com/PharMolix/MutaPLM)

* **ProtDAT: A Unified Framework for Protein Sequence Design from Any Protein Text Description**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2412.04069)
  [![Stars](https://img.shields.io/github/stars/GXY0116/ProtDAT?color=yellow&style=social)](https://github.com/GXY0116/ProtDAT)

* **EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2412.11618)
  [![Stars](https://img.shields.io/github/stars/sornkL/EvoLlama?color=yellow&style=social)](https://github.com/sornkL/EvoLlama)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/nwliu/EvoLlama)

* **FAPM: functional annotation of proteins using multimodal models beyond structural modeling**
  
  [![](https://img.shields.io/badge/Bioinformatics_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://academic.oup.com/bioinformatics/article/40/12/btae680/7900294)
  [![Stars](https://img.shields.io/github/stars/xiangwenkai/FAPM?color=yellow&style=social)](https://github.com/xiangwenkai/FAPM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/wenkai/FAPM/tree/main/model)

* **ProCyon: A multimodel foundation model for protein phenotypes**
  
  [![](https://img.shields.io/badge/bioRxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1)
  [![Stars](https://img.shields.io/github/stars/mims-harvard/ProCyon?color=yellow&style=social)](https://github.com/mims-harvard/ProCyon)

* **(Evolla) Decoding the Molecular Language of Proteins with Evolla**
  
  [![](https://img.shields.io/badge/bioRxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2025.01.05.630192v1)
  [![Stars](https://img.shields.io/github/stars/westlake-repl/Evolla?color=yellow&style=social)](https://github.com/westlake-repl/Evolla)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/westlake-repl/Evolla-10B)

* **ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding**
  
  [![](https://img.shields.io/badge/ICLR_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=qWlqfjGVWX)
  [![Stars](https://img.shields.io/github/stars/OviaLabs/ProteinGPT?color=yellow&style=social)](https://github.com/OviaLabs/ProteinGPT)

* **(MP4) A generalized protein design ML model enables generation of functional de novo proteins**
  
  [![](https://img.shields.io/badge/ICLR_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=H7KaafflRG)

* **Protclip: Function-informed protein multi-modal learning**
  
  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/34456)
  [![Stars](https://img.shields.io/github/stars/diaoshaoyou/ProtCLIP?color=yellow&style=social)](https://github.com/diaoshaoyou/ProtCLIP)

* **(CtrlProt) Controllable Protein Sequence Generation with LLM Preference Optimization Authors**
  
  [![](https://img.shields.io/badge/AAAI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/32030)
  [![Stars](https://img.shields.io/github/stars/nju-websoft/CtrlProt?color=yellow&style=social)](https://github.com/nju-websoft/CtrlProt)

* **Protein2Text: Resampling Mechanism to Translate Protein Sequences into Human-Interpretable Text**
  
  [![](https://img.shields.io/badge/NAACL_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2025.naacl-industry.68/)
  [![Stars](https://img.shields.io/github/stars/alaaj27/Protein2Text?color=yellow&style=social)](https://github.com/alaaj27/Protein2Text)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M)

* **InstructPro: Natural Language Guided Ligand-Binding Protein Design**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2506.09332)

* **(RAPM): Rethinking Text-based Protein Understanding: Retrieval or LLM?**
  
  [![](https://img.shields.io/badge/bioRxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.20354)
  [![Stars](https://img.shields.io/github/stars/IDEA-XL/RAPM?color=yellow&style=social)](https://github.com/IDEA-XL/RAPM)

* **Prot2Text-V2: Protein Function Prediction with Multimodal Contrastive Alignment**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.11194)
  [![Stars](https://img.shields.io/github/stars/colinfx/prot2text-v2?color=yellow&style=social)](https://github.com/colinfx/prot2text-v2)

* **(ProDVa) Protein Design with Dynamic Protein Vocabulary**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2505.18966)
  [![Stars](https://img.shields.io/github/stars/sornkL/ProDVa?color=yellow&style=social)](https://github.com/sornkL/ProDVa)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/collections/nwliu/prodva)

* **ProteinAligner: A Tri-Modal Contrastive Learning Framework for Protein Representation Learning**
  
  [![](https://img.shields.io/badge/ICML_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=LjlJBnmZ0M)
  [![Stars](https://img.shields.io/github/stars/Alexiland/ProteinAligner?color=yellow&style=social)](https://github.com/Alexiland/ProteinAligner)

* **Prottex: Structure-in-context reasoning and editing of proteins with large language models**
  
  [![](https://img.shields.io/badge/JCIM_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c00585)
  [![Stars](https://img.shields.io/github/stars/mzc2113391/ProtTeX?color=yellow&style=social)](https://github.com/mzc2113391/ProtTeX)

* **Prot2Chat: protein large language model with early fusion of text, sequence, and structure**
  
  [![](https://img.shields.io/badge/Bioinformatics_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://academic.oup.com/bioinformatics/article/41/8/btaf396/8215464)
  [![Stars](https://img.shields.io/github/stars/wangzc1233/Prot2Chat?color=yellow&style=social)](https://github.com/wangzc1233/Prot2Chat)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zcw51699/prot2chat/tree/main/zcw51699/prot2chat)
  
* **(CLASP) Multi-Modal Protein Representation Learning with CLASP**
  
  [![](https://img.shields.io/badge/bioRxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2025.08.10.669533v1.full.pdf)
  [![Stars](https://img.shields.io/github/stars/Emad-COMBINE-lab/clasp?color=yellow&style=social)](https://github.com/Emad-COMBINE-lab/clasp)

* **ProTrek: Navigating the Protein Universe through Tri-Modal Contrastive Learning**
  
  [![](https://img.shields.io/badge/Nature_Biotechnology_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41587-025-02836-0)
  [![Stars](https://img.shields.io/github/stars/westlake-repl/ProTrek?color=yellow&style=social)](https://github.com/westlake-repl/ProTrek)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50)

* **Protein as a Section Language for LLMs**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2510.11188)

* **Caduceus: MoE-enhanced Foundation Models Unifying Biological and Natural Language**
  
  [![](https://img.shields.io/badge/ICLR_Submission_2026-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/forum?id=NNRQW01Xuh)

### More Modalities

* **Galactica: A Large Language Model for Science**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://galactica.org/static/paper.pdf)
  [![Stars](https://img.shields.io/github/stars/paperswithcode/galai?color=yellow&style=social)](https://github.com/paperswithcode/galai)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/models?other=galactica)

* **BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.70.pdf)
  [![Stars](https://img.shields.io/github/stars/QizhiPei/BioT5?color=yellow&style=social)](https://github.com/QizhiPei/BioT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/QizhiPei/biot5-base)

* **DARWIN Series: Domain Specific Large Language Models for Natural Science**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.13565.pdf)
  [![Stars](https://img.shields.io/github/stars/MasterAI-EAM/Darwin?color=yellow&style=social)](https://github.com/MasterAI-EAM/Darwin)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/MasterAI-EAM/Darwin)

* **BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.09442.pdf)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://pan.baidu.com/share/init?surl=iAMBkuoZnNAylhopP5OgEg\&pwd=7a6b)

* **(StructChem) Structured Chemistry Reasoning with Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.09656.pdf)
  [![Stars](https://img.shields.io/github/stars/ozyyshr/StructChem?color=yellow&style=social)](https://github.com/ozyyshr/StructChem?tab=readme-ov-file)

* **(BioTranslator) Multilingual translation for zero-shot biomedical classification using BioTranslator**
  
  [![](https://img.shields.io/badge/Nature_Communications_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41467-023-36476-2.pdf)
  [![Stars](https://img.shields.io/github/stars/HanwenXuTHU/BioTranslatorProject?color=yellow&style=social)](https://github.com/HanwenXuTHU/BioTranslatorProject)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://figshare.com/articles/dataset/Protein_Pathway_data_tar/20120447)

* **Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=Tlsdsb6l9n)
  [![Stars](https://img.shields.io/github/stars/zjunlp/Mol-Instructions?color=yellow&style=social)](https://github.com/zjunlp/Mol-Instructions)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp)

* **(ChatDrug) ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=yRrPfKyJQ2)
  [![Stars](https://img.shields.io/github/stars/chao1224/ChatDrug?color=yellow&style=social)](https://github.com/chao1224/ChatDrug)

* **BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=jJCeMiwHdH)
  [![Stars](https://img.shields.io/github/stars/RyanWangZf/BioBridge?color=yellow&style=social)](https://github.com/RyanWangZf/BioBridge)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/RyanWangZf/BioBridge/tree/main/checkpoints)

* **(KEDD) Towards Unified AI Drug Discovery with Multiple Knowledge Modalities**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.01523.pdf)

* **（Otter Knowledge) Knowledge Enhanced Representation Learning for Drug Discovery**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ojs.aaai.org/index.php/AAAI/article/view/28924)
  [![Stars](https://img.shields.io/github/stars/IBM/otter-knowledge?color=yellow&style=social)](https://github.com/IBM/otter-knowledge)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/ibm)

* **ChatCell: Facilitating Single-Cell Analysis with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.08303v2.pdf)
  [![Stars](https://img.shields.io/github/stars/zjunlp/ChatCell?color=yellow&style=social)](https://github.com/zjunlp/ChatCell)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp)

* **LangCell: Language-Cell Pre-training for Cell Identity Understanding**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2405.06708v2)

* **BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning**
  
  [![](https://img.shields.io/badge/ACL_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.17810.pdf)
  [![Stars](https://img.shields.io/github/stars/QizhiPei/BioT5?color=yellow&style=social)](https://github.com/QizhiPei/BioT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/QizhiPei/BioT5)

* **MolBind: Multimodal Alignment of Language, Molecules, and Proteins**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.08167.pdf)
  [![Stars](https://img.shields.io/github/stars/tengxiao1/MolBind?color=yellow&style=social)](https://github.com/tengxiao1/MolBind)

* **Uni-SMART: Universal Science Multimodal Analysis and Research Transformer**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.10301.pdf)

* **Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.05140.pdf)
  [![Stars](https://img.shields.io/github/stars/sjunhongshen/Tag-LLM?color=yellow&style=social)](https://github.com/sjunhongshen/Tag-LLM)

* **An Evaluation of Large Language Models in Bioinformatics Research**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.13714.pdf)

* **SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences**
  
  [![](https://img.shields.io/badge/bioRxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2024.07.16.603812v1.full.pdf)

* **SciDFM: A Large Language Model with Mixture-of-Experts for Science**
  
  [![](https://img.shields.io/badge/NeurIPS_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2409.18412)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/OpenDFM/SciDFM-MoE-A5.6B-v1.0)

* **ChemDFM-X: Towards Large Multimodal Model for Chemistry**
  
  [![](https://img.shields.io/badge/Sci_China_Inf_Sci_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2409.13194)
  [![Stars](https://img.shields.io/github/stars/OpenDFM/ChemDFM-X?color=yellow&style=social)](https://github.com/OpenDFM/ChemDFM-X)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/OpenDFM/ChemDFM-X-v1.0-13B)

* **(NatureLM) Nature Language Model: Deciphering the Language of Nature for Scientific Discovery**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2502.07527)
  [![Stars](https://img.shields.io/github/stars/microsoft/SFM?color=yellow&style=social)](https://github.com/microsoft/SFM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/collections/microsoft/naturelm)

* **(ChemDFM) Developing ChemDFM as a large language foundation model for chemistry**
  
  [![](https://img.shields.io/badge/Cell_Rep_Phys_Sci_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(25)00122-5)
  [![Stars](https://img.shields.io/github/stars/OpenDFM/ChemDFM?color=yellow&style=social)](https://github.com/OpenDFM/ChemDFM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/OpenDFM/ChemDFM-v1.0-13B)

* **(KFPPIMI) Improving protein–protein interaction modulator predictions via knowledge-fused language models**
  
  [![](https://img.shields.io/badge/Information_Fusion_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.sciencedirect.com/science/article/abs/pii/S1566253525003008?via%3Dihub)
  [![Stars](https://img.shields.io/github/stars/1zzt/KFPPIMI?color=yellow&style=social)](https://github.com/1zzt/KFPPIMI)

* **(CAFT) Improving Large Language Models with  Concept-Aware Fine-Tuning**
  
  [![](https://img.shields.io/badge/Information_Fusion_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2506.07833)
  [![Stars](https://img.shields.io/github/stars/michaelchen-lab/caft-llm?color=yellow&style=social)](https://github.com/michaelchen-lab/caft-llm)

* **(InstructBioMol) Advancing biomolecular understanding and design following human instructions**
  
  [![](https://img.shields.io/badge/NMI_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s42256-025-01064-0)
  [![Stars](https://img.shields.io/github/stars/HICAI-ZJU/InstructBioMol?color=yellow&style=social)](https://github.com/HICAI-ZJU/InstructBioMol)

* **DrugLM: A Unified Framework to Enhance Drug-Target Interaction Predictions by Incorporating Textual Embeddings via Language Models**
  
  [![](https://img.shields.io/badge/bioRxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2025.07.09.657250v1.abstract)
  [![Stars](https://img.shields.io/github/stars/HICAI-ZJU/InstructBioMol?color=yellow&style=social)](https://github.com/HICAI-ZJU/InstructBioMol)

* **Intern-S1: A Scientific Multimodal Foundation Model**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2508.15763)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/internlm/Intern-S1)

* **Chem3DLLM: 3D Multimodal Large Language Models for Chemistry**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2508.10696)

* **SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines**
  
  [![](https://img.shields.io/badge/Arxiv_2025-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2509.21320)
  [![Stars](https://img.shields.io/github/stars/open-sciencelab/SciReason?color=yellow&style=social)](https://github.com/open-sciencelab/SciReason)

## Datasets & Benchmarks
| **Dataset**          | **Usage**         | **Modality**            | **Link**                                                             |
|----------------------|-------------------|-------------------------|----------------------------------------------------------------------|
| PubMed               | Pre-training      | Text                    | [https://pubmed.ncbi.nlm.nih.gov/download](https://pubmed.ncbi.nlm.nih.gov/download) |
| bioRxiv              | Pre-training      | Text                    | [https://huggingface.co/datasets/mteb/raw_biorxiv](https://huggingface.co/datasets/mteb/raw_biorxiv),[https://www.biorxiv.org/tdm](https://www.biorxiv.org/tdm) |
| MedRxiv              | Pre-training      | Text                    | [https://www.medrxiv.org/tdm](https://www.medrxiv.org/tdm)           |
| S2ORC                | Pre-training      | Text                    | [https://github.com/allenai/s2orc](https://github.com/allenai/s2orc) |
| MIMIC                | Pre-training      | Text                    | [https://physionet.org/content/mimiciii/1.4](https://physionet.org/content/mimiciii/1.4) |
| UF Health            | Pre-training      | Text                    | [https://idr.ufhealth.org](https://idr.ufhealth.org)                 |
| Elsevier Corpus      | Pre-training      | Text                    | [https://elsevier.digitalcommonsdata.com/datasets/zm33cdndxs/3](https://elsevier.digitalcommonsdata.com/datasets/zm33cdndxs/3) |
| Eurpoe PMC           | Pre-training      | Text                    | [https://europepmc.org/downloads](https://europepmc.org/downloads)   |
| LibreText            | Pre-training      | Text                    | [https://chem.libretexts.org](https://chem.libretexts.org/)          |
| NLM literature archive | Pre-training    | Text                    | [https://ftp.ncbi.nlm.nih.gov/pub/litarch/](https://ftp.ncbi.nlm.nih.gov/pub/litarch/) |
| GAP-Replay           | Pre-training      | Text                    | -                                                                    |
| ZINC                 | Pre-training      | Molecule                | [https://zinc15.docking.org](https://zinc15.docking.org), [https://zinc20.docking.org](https://zinc20.docking.org) |
| UniProt              | Pre-training      | Protein                 | [https://www.uniprot.org](https://www.uniprot.org)                   |
| ChEMBL               | Pre-training      | Molecule, Bioassay      | [https://www.ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl)         |
| GIMLET               | Pre-training      | Molecule, Bioassay      | [https://github.com/zhao-ht/GIMLET](https://github.com/zhao-ht/GIMLET), [https://huggingface.co/datasets/haitengzhao/molecule_property_instruction](https://huggingface.co/datasets/haitengzhao/molecule_property_instruction) |
| PubChem              | Pre-training      | Text, Molecule, IUPAC, etc | [https://ftp.ncbi.nlm.nih.gov/pubchem](https://ftp.ncbi.nlm.nih.gov/pubchem) |
| InterPT              | Pre-training      | Text, Protein           | [https://huggingface.co/datasets/ProtLLM/ProtLLM](https://huggingface.co/datasets/ProtLLM/ProtLLM) |
| STRING               | Pre-training      | Text, Protein, etc      | [https://string-db.org](https://string-db.org)                       |
| MolTextNet               | Pre-training      | Text, Molecule    | [https://huggingface.co/datasets/liuganghuggingface/moltextnet](https://huggingface.co/datasets/liuganghuggingface/moltextnet)                       |
| BLURB                | Fine-tuning       | Text                    | [https://microsoft.github.io/BLURB](https://microsoft.github.io/BLURB) |
| PubMedQA             | Fine-tuning       | Text                    | [https://github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) |
| SciQ                 | Fine-tuning       | Text                    | [https://huggingface.co/datasets/sciq](https://huggingface.co/datasets/sciq) |
| BioASQ               | Fine-tuning       | Text                    | [http://participants-area.bioasq.org/datasets](http://participants-area.bioasq.org/datasets) |
| MoleculeNet          | Fine-tuning       | Molecule                | [https://moleculenet.org/datasets-1](https://moleculenet.org/datasets-1) |
| MoleculeACE          | Fine-tuning       | Molecule                | [https://github.com/molML/MoleculeACE](https://github.com/molML/MoleculeACE) |
| TDC                  | Fine-tuning       | Molecule                | [https://tdcommons.ai/](https://tdcommons.ai/)                        |
| USPTO                | Fine-tuning       | Molecule                | [https://yzhang.hpc.nyu.edu/T5Chem](https://yzhang.hpc.nyu.edu/T5Chem) |
| Graph2graph          | Fine-tuning       | Molecule                | [https://github.com/wengong-jin/iclr19-graph2graph/tree/master/data](https://github.com/wengong-jin/iclr19-graph2graph/tree/master/data) |
| PEER                 | Fine-tuning       | Protein                 | [https://github.com/DeepGraphLearning/PEER_Benchmark](https://github.com/DeepGraphLearning/PEER_Benchmark) |
| FLIP                 | Fine-tuning       | Protein                 | [https://benchmark.protein.properties](https://benchmark.protein.properties) |
| TAPE                 | Fine-tuning       | Protein                 | [https://github.com/songlab-cal/tape](https://github.com/songlab-cal/tape) |
| PubChemSTM           | Fine-tuning       | Text, Molecule          | [https://huggingface.co/datasets/chao1224/MoleculeSTM/tree/main](https://huggingface.co/datasets/chao1224/MoleculeSTM/tree/main) |
| PseudoMD-1M          | Fine-tuning       | Text, Molecule          | [https://huggingface.co/datasets/SCIR-HI/PseudoMD-1M](https://huggingface.co/datasets/SCIR-HI/PseudoMD-1M) |
| ChEBI-20             | Fine-tuning       | Text, Molecule          | [https://github.com/blender-nlp/MolT5](https://github.com/blender-nlp/MolT5)|
| ChEBI-20-MM          | Fine-tuning       | Text, Molecule          | [https://github.com/AI-HPC-Research-Team/SLM4Mol](https://github.com/AI-HPC-Research-Team/SLM4Mol) |
| ChEBL-dia            | Fine-tuning       | Text, Molecule          | [https://github.com/Ellenzzn/ChatMol/tree/main/data/ChEBI-dia](https://github.com/Ellenzzn/ChatMol/tree/main/data/ChEBI-dia) |
| L+M-24               | Fine-tuning       | Text, Molecule          | [https://github.com/language-plus-molecules/LPM-24-Dataset](https://github.com/language-plus-molecules/LPM-24-Dataset) |
| PCdes                | Fine-tuning       | Text, Molecule          | [https://github.com/thunlp/KV-PLM](https://github.com/thunlp/KV-PLM)|
| MoMu                 | Fine-tuning       | Text, Molecule          | [https://github.com/yangzhao1230/GraphTextRetrieval](https://github.com/yangzhao1230/GraphTextRetrieval) |
| PubChemQA            | Fine-tuning       | Text, Molecule          | [https://github.com/PharMolix/OpenBioMed](https://github.com/PharMolix/OpenBioMed) |
| 3D-MolT              | Fine-tuning       | Text, Molecule          | [https://huggingface.co/datasets/Sihangli/3D-MoIT](https://huggingface.co/datasets/Sihangli/3D-MoIT) |
| ChemCoTDataset           | Fine-tuning       | Text, Molecule          | [https://huggingface.co/datasets/OpenMol/ChemCoTDataset](https://huggingface.co/datasets/OpenMol/ChemCoTDataset) |
| MoleculeQA           | Fine-tuning       | Text, Molecule          | [https://github.com/IDEA-XL/MoleculeQA](https://github.com/IDEA-XL/MoleculeQA) |
| MolTextQA           | Fine-tuning       | Text, Molecule          | [https://github.com/siddharthal/MolTextQA](https://github.com/siddharthal/MolTextQA) |
| DrugBank             | Fine-tuning       | Text, Molecule, etc     | [https://github.com/SCIR-HI/ArtificiallyR2R](https://github.com/SCIR-HI/ArtificiallyR2R)|
| SwissProt            | Fine-tuning       | Text, Protein           | [https://www.expasy.org/resources/uniprotkb-swiss-prot](https://www.expasy.org/resources/uniprotkb-swiss-prot)|
| UniProtQA            | Fine-tuning       | Text, Protein           | [https://github.com/PharMolix/OpenBioMed](https://github.com/PharMolix/OpenBioMed) |
| SciEval              | Instruction       | Text                    | [https://github.com/OpenDFM/SciEval](https://github.com/OpenDFM/SciEval) |
| BioInfo-Bench        | Instruction       | Text                    | [https://github.com/cinnnna/bioinfo-bench](https://github.com/cinnnna/bioinfo-bench) |
| MedC-I               | Instruction       | Text                    | [https://huggingface.co/datasets/axiong/pmc_llama_instructions](https://huggingface.co/datasets/axiong/pmc_llama_instructions) |
| BioMedEval           | Instruction       | Text                    | [https://github.com/tahmedge/llm-eval-biomed](https://github.com/tahmedge/llm-eval-biomed) |
| MolOpt-Instructions  | Instruction       | Text, Molecule          | [https://github.com/blazerye/DrugAssist](https://github.com/blazerye/DrugAssist) |
| SMolInstruct         | Instruction       | Text, Molecule          | [https://github.com/OSU-NLP-Group/LLM4Chem](https://github.com/OSU-NLP-Group/LLM4Chem) |
| ChemLLMBench         | Instruction       | Text, Molecule          | [https://github.com/ChemFoundationModels/ChemLLMBench](https://github.com/ChemFoundationModels/ChemLLMBench) |
| AI4Chem              | Instruction       | Text, Molecule          | [https://github.com/andresilvapimentel/AI4Chem](https://github.com/andresilvapimentel/AI4Chem) |
| GPTChem              | Instruction       | Text, Molecule          | [https://github.com/kjappelbaum/gptchem](https://github.com/kjappelbaum/gptchem)|
| OpenMolIns     | Instruction       | Text, Molecule          | [https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench) |
| SLM4CRP_with_RTs     | Instruction       | Text, Molecule          | [https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs](https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs) |
| DARWIN               | Instruction       | Text, Molecule, etc     | [https://github.com/MasterAI-EAM/Darwin/tree/main/dataset](https://github.com/MasterAI-EAM/Darwin/tree/main/dataset) |
| StructChem           | Instruction       | Text, Molecule, etc     | [https://github.com/ozyyshr/StructChem](https://github.com/ozyyshr/StructChem) |
| SciAssess            | Instruction       | Text, Molecule, etc     | [https://sci-assess.github.io](https://sci-assess.github.io/), [https://github.com/sci-assess/SciAssess](https://github.com/sci-assess/SciAssess) |
| InstructProtein      | Instruction       | Text, Protein           | - |
| Open Protein Instructions | Instruction | Text, Protein          | [https://github.com/baaihealth/opi](https://github.com/baaihealth/opi) |
| ProteinLMDataset             | Instruction                 | Text, Protein | [https://huggingface.co/datasets/tsynbio/ProteinLMDataset](https://huggingface.co/datasets/tsynbio/ProteinLMDataset)|
| OPI             | Instruction                 | Text, Protein | [https://github.com/baaihealth/opi](https://github.com/baaihealth/opi)|
| Mol-Instructions     | Instruction       | Text, Molecule, Protein | [https://huggingface.co/datasets/zjunlp/Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |
| Biology-Instructions             | Instruction                 | Text, Molecule, Protein, etc | [https://github.com/hhnqqq/BiologyInstructions](https://github.com/hhnqqq/BiologyInstructions)|
| CheF                 | -                 | Text, Molecule          | [https://github.com/kosonocky/CheF](https://github.com/kosonocky/CheF) |
| ChemCoTBench                 | -                 | Text, Molecule          | [https://huggingface.co/datasets/OpenMol/ChemCoTDataset](https://huggingface.co/datasets/OpenMol/ChemCoTDataset) |
| MolCap-Arena                 | -                 | Text, Molecule          | [https://github.com/Genentech/molcap-arena](https://github.com/Genentech/molcap-arena) |
| MolErr2Fix                 | -                 | Text, Molecule          | [https://huggingface.co/datasets/YoungerWu/MolErr2Fix](https://huggingface.co/datasets/YoungerWu/MolErr2Fix) |
| MolLangBench                 | -                 | Text, Molecule          | [https://huggingface.co/datasets/ChemFM/MolLangBench](https://huggingface.co/datasets/ChemFM/MolLangBench) |
| IUPAC Gold Book      | -                 | Text, Molecule          | [https://goldbook.iupac.org](https://goldbook.iupac.org/) |
| ChemNLP              | -                 | Text, Molecule, etc     | [https://github.com/OpenBioML/chemnlp](https://github.com/OpenBioML/chemnlp) |
| ChemFOnt             | -                 | Text, Molecule, Protein, etc | [https://www.chemfont.ca](https://www.chemfont.ca)|
| ProteinLMBench             | -                 | Text, Protein | [https://huggingface.co/datasets/tsynbio/ProteinLMBench](https://huggingface.co/datasets/tsynbio/ProteinLMBench)|


## Related Resources
### Related Surveys & Evaluations
* A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery [Arxiv 2406](https://arxiv.org/pdf/2406.10833)
* Bridging Text and Molecule: A Survey on Multimodal Frameworks for Molecule [Arxiv 2403](https://arxiv.org/abs/2403.13830)
* Bioinformatics and Biomedical Informatics with ChatGPT: Year One Review [Arxiv 2403](https://arxiv.org/abs/2403.15274)
* From Words to Molecules: A Survey of Large Language Models in Chemistry [Arxiv 2402](https://arxiv.org/abs/2402.01439)
* Scientific Language Modeling: A Quantitative Review of Large Language Models in Molecular Science [Arxiv 2402](https://arxiv.org/abs/2402.04119)
* Progress and Opportunities of Foundation Models in Bioinformatics [Arxiv 2402](https://arxiv.org/abs/2402.04286)
* Scientific Large Language Models: A Survey on Biological & Chemical Domains [Arxiv 2401](https://arxiv.org/abs/2401.14656)
* The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 [Arxiv 2311](https://arxiv.org/abs/2311.07361)
* Transformers and Large Language Models for Chemistry and Drug Discovery [Arxiv 2310](https://arxiv.org/abs/2310.06083)
* Language models in molecular discovery [Arxiv 2309](https://arxiv.org/abs/2309.16235)
* What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks [NeurIPS 2309](https://openreview.net/pdf?id=1ngbR3SZHW)
* Do Large Language Models Understand Chemistry? A Conversation with ChatGPT [JCIM 2303](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00285)
* A Systematic Survey of Chemical Pre-trained Models [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0760.pdf)

### Related Workshop
* [Language + Molecules @ ACL 2024 Workshop](https://language-plus-molecules.github.io/)

### Related Repositories
* [LLM4ScientificDiscovery](https://github.com/microsoft/LLM4ScientificDiscovery)
* [SLM4Mol](https://github.com/AI-HPC-Research-Team/SLM4Mol)
* [Scientific-LLM-Survey](https://github.com/HICAI-ZJU/Scientific-LLM-Survey)
* [Awesome-Bio-Foundation-Models](https://github.com/apeterswu/Awesome-Bio-Foundation-Models)
* [Awesome-Molecule-Text](https://github.com/Namkyeong/awesome-molecule-text)
* [LLM4Mol](https://github.com/HHW-zhou/LLM4Mol)
* [Awesome-Chemical-Pre-trained-Models](https://github.com/junxia97/awesome-pretrain-on-molecules)
* [Awesome-Chemistry-Datasets](https://github.com/kjappelbaum/awesome-chemistry-datasets)
* [Awesome-Docking](https://github.com/KyGao/awesome-docking)

## Acknowledgements
This repository is contributed and updated by [QizhiPei](https://qizhipei.github.io) and [Lijun Wu](https://apeterswu.github.io). If you have questions, don't hesitate to open an issue or ask me via <qizhipei@ruc.edu.cn> or Lijun Wu via <lijun_wu@outlook.com>. We are happy to hear from you!

## Citations
```
@article{pei2024leveraging,
  title={Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey},
  author={Pei, Qizhi and Wu, Lijun and Gao, Kaiyuan and Zhu, Jinhua and Wang, Yue and Wang, Zun and Qin, Tao and Yan, Rui},
  journal={arXiv preprint arXiv:2403.01528},
  year={2024}
}
```

![Star History Chart](https://api.star-history.com/svg?repos=QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling&type=Date)
