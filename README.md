<h1 align="center">
🧬📝 Awesome Biomolecule-Language Cross Modeling
</h1>
<div align="center">

[![](https://img.shields.io/badge/paper-arxiv2403.01528-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2403.01528)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling?color=yellow&labelColor=555555)  ![Forks](https://img.shields.io/github/forks/QizhiPei/Awesome-Biomolecule-Language-Cross-Modeling?color=blue&label=Fork&labelColor=555555)
</div>

The repository for [Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey](https://arxiv.org/abs/2403.01528), including related models, datasets/benchmarks, and other resource links.

🌟 **If you have a paper or resource you'd like to add, feel free to submit a pull request or open an issue.**

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
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1e43c7084bdcb6b3102afaf301cce10faead2702%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/dmis-lab/biobert?color=yellow&style=social)](https://github.com/dmis-lab/biobert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/dmis-lab)

* **SciBERT: A Pretrained Language Model for Scientific Text**
  
  [![](https://img.shields.io/badge/EMNLP_IJCNLP_2019-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/D19-1371.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F156d217b0a911af97fa1b5a71dc909ccef7a8028%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/allenai/scibert?color=yellow&style=social)](https://github.com/allenai/scibert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/allenai/scibert)

* **(BlueBERT) Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2019-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/W19-5006.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F347bac45298f37cd83c3e79d99b826dc65a70c46%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ncbi-nlp/bluebert?color=yellow&style=social)](https://github.com/ncbi-nlp/bluebert)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/bionlp)

* **Bio-Megatron: Larger Biomedical Domain Language Model**
  
  [![](https://img.shields.io/badge/EMNLP_2020-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2020.emnlp-main.379.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F347bac45298f37cd83c3e79d99b826dc65a70c46%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/NeMo?color=yellow&style=social)](https://github.com/NVIDIA/NeMo)

* **ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission**
  
  [![](https://img.shields.io/badge/BioNLP@CHIL_2020-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/1904.05342.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb3c2c9f53ab130f3eb76eaaab3afa481c5a405eb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/kexinhuang12345/clinicalBERT?color=yellow&style=social)](https://github.com/kexinhuang12345/clinicalBERT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/lindvalllab/clinicalXLNet)

* **BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2021.bionlp-1.24.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9d2a0337979c62cb6ce337d9977dd9b2020da98d%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/salrowili/BioM-Transformers?color=yellow&style=social)](https://github.com/salrowili/BioM-Transformers)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/sultan)

* **(PubMedBERT) Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing**
  
  [![](https://img.shields.io/badge/HEALTH_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://dl.acm.org/doi/10.1145/3458754)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa2f38d03fd363e920494ad65a5f0ad8bd18cd60b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

* **SciFive: a text-to-text transformer model for biomedical literature**
  
  [![](https://img.shields.io/badge/Arxiv_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2106.03598.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6003d268e9b5230dbc3e320497b50329d6186816%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/justinphan3110/SciFive?color=yellow&style=social)](https://github.com/justinphan3110/SciFive)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/razent)

* **(DRAGON) Deep Bidirectional Language-Knowledge Graph Pretraining**
  
  [![](https://img.shields.io/badge/NeurIPS_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2210.09338.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fad3dfb2514cb0c899fcb9a14d229ff2a6018892f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/michiyasunaga/dragon?color=yellow&style=social)](https://github.com/michiyasunaga/dragon)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/michiyasunaga/dragon)

* **LinkBERT: Pretraining Language Models with Document Links**
  
  [![](https://img.shields.io/badge/ACL_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.acl-long.551.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa83cdcc0135c58fddf89fc72f1b92b7a9d1e170f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/michiyasunaga/LinkBERT?color=yellow&style=social)](https://github.com/michiyasunaga/LinkBERT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/michiyasunaga)

* **BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model**
  
  [![](https://img.shields.io/badge/BioNLP@ACL_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.bionlp-1.9.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0db5207510819b9956849eb84bfe8703f8f3688d%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/GanjinZero/BioBART?color=yellow&style=social)](https://github.com/GanjinZero/BioBART)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GanjinZero)

* **BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining**
  
  [![](https://img.shields.io/badge/Bioinformatics_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1093/bib/bbac409)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F44279244407a64431810f982be6d0c7da4429dd7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/microsoft/BioGPT?color=yellow&style=social)](https://github.com/microsoft/BioGPT)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/microsoft/biogpt)

* **GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2203.03540.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Faf0fe7f31315a4173de5695887dffb20be458f48%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM?color=yellow&style=social)](https://github.com/NVIDIA/Megatron-LM)

* **Large language models encode clinical knowledge**
  
  [![](https://img.shields.io/badge/Nature_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41586-023-06291-2.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6052486bc9144dc1730c12bf35323af3792a1fd0%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **(ScholarBERT) The Diminishing Returns of Masked Language Models to Science**
  
  [![](https://img.shields.io/badge/ACL_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.findings-acl.82.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F44279244407a64431810f982be6d0c7da4429dd7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/globuslabs/ScholarBERT)

* **PMC-LLaMA: Further Finetuning LLaMA on Medical Papers**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.14454.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd2d1a0f73ba8c40f882a386cd367899785fb877%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/chaoyi-wu/PMC-LLaMA?color=yellow&style=social)](https://github.com/chaoyi-wu/PMC-LLaMA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/chaoyi-wu/PMC-LLaMA)


* **BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.09442v2.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe3fd89a7f6b28973cfc68bfc51caebd8fb93f0bc%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)

* **(GatortronGPT) A study of generative large language model for medical research and healthcare**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.13523.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe154dd91de91558f9d671370754eace62a54c911%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM?color=yellow&style=social)](https://github.com/NVIDIA/Megatron-LM)

* **Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.12031.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F61b0f5cfd4f951632435707948201474e16e835b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/bowang-lab/clinical-camel?color=yellow&style=social)](https://github.com/bowang-lab/clinical-camel)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/wanglab/ClinicalCamel-70B)

* **MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.16079.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fff5f0c5b6905a8c4b361a625b450e9ab417fa854%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/epfLLM/meditron?color=yellow&style=social)](https://github.com/epfLLM/meditron)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/epfl-llm/meditron-70b)

* **BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-inspired Materials**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.08788.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8db921900955a447d389582143912eee3046fd3e%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/lamm-mit/BioinspiredLLM)

* **ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.09968.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Febc502a4d173f6550a8cd6384cb06f2c43c7c1a3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/medicalai/ClinicalGPT-base-zh)

* **MedAlpaca - An Open-Source Collection of Medical Conversational AI Models and Training Data**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.08247.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F38a9609a5bd874534527df9b00f2897927e57be9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/kbressem/medAlpaca?color=yellow&style=social)](https://github.com/kbressem/medAlpaca)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/medalpaca)

* **SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.07950.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc6e162aedf6a5ab0135e3b991577d77ca06673f9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/THUDM/SciGLM?color=yellow&style=social)](https://github.com/THUDM/SciGLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zd21/SciGLM-6B)

* **BioMedLM**
  
  [![](https://img.shields.io/badge/Report-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://crfm.stanford.edu/2022/12/15/biomedlm.html)
  [![Stars](https://img.shields.io/github/stars/stanford-crfm/BioMedLM?color=yellow&style=social)](https://github.com/stanford-crfm/BioMedLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/stanford-crfm/BioMedLM)


### Text + Molecule
* **Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries**
  
  [![](https://img.shields.io/badge/EMNLP_2021-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2021.emnlp-main.47.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F57651d65078818821234d13544ac1f29858dcd67%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/cnedwards/text2mol?color=yellow&style=social)](https://github.com/cnedwards/text2mol)
  

* **(MolT5) Translation between Molecules and Natural Language**  
  
  [![](https://img.shields.io/badge/EMNLP_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2022.emnlp-main.26.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3b9b1aba877ecd3f7e508cbc78a41b623349902b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/blender-nlp/MolT5?color=yellow&style=social)](https://github.com/blender-nlp/MolT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/laituan245)

* **(KV-PLM) A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals**
  
  [![](https://img.shields.io/badge/Nature_Communications_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41467-022-28494-3.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6958612fea7f220757b4165b8e12d4b62b4baa80%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/thunlp/KV-PLM?color=yellow&style=social)](https://github.com/thunlp/KV-PLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX)

* **(MoMu) A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2209.05481.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1c7a4e8d9f4fcf19a5d1caa078c66ca39cb75dd2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/BingSu12/MoMu?color=yellow&style=social)](https://github.com/BingSu12/MoMu)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/BingSu12/MoMu)

* **(Text+Chem T5) Unifying Molecular and Textual Representations via Multi-task Language Modelling**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/christofidellis23a/christofidellis23a.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb822f2abca1da6f990b2bd47ed43da0671bfc6f8%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/GT4SD/multitask_text_and_chemistry_t5?color=yellow&style=social)](https://github.com/GT4SD/multitask_text_and_chemistry_t5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GT4SD)

* **(CLAMP) Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/seidl23a/seidl23a.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F30809168fff23c852867ad359baaebfae532f0a7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ml-jku/clamp?color=yellow&style=social)](https://github.com/ml-jku/clamp)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/ml-jku/clamp)

* **GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning**
  
  [![](https://img.shields.io/badge/NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=Tt6DrRCgJV)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F119a3ed0898499fce0ce6af6958d566d82390ba5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/zhao-ht/GIMLET?color=yellow&style=social)](https://github.com/zhao-ht/GIMLET)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/haitengzhao/gimlet)

* **(HI-Mol) Data-Efficient Molecular Generation with Hierarchical Textual Inversion**
  
  [![](https://img.shields.io/badge/AI4D3@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=wwotGBxtC3)

* **MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction**
  
  [![](https://img.shields.io/badge/AI4D3@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ai4d3.github.io/papers/34.pdf)

* **(ChemLLMBench) What indeed can GPT models do in chemistry? A comprehensive benchmark on eight tasks**
  
  [![](https://img.shields.io/badge/Datasets&Benchmarks@NeurIPS_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=1ngbR3SZHW)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F119a3ed0898499fce0ce6af6958d566d82390ba5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ChemFoundationModels/ChemLLMBench?color=yellow&style=social)](https://github.com/ChemFoundationModels/ChemLLMBench)

* **MolXPT: Wrapping Molecules with Text for Generative Pre-training**
  
  [![](https://img.shields.io/badge/ACL_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.acl-short.138.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fed1353d705eeabc0e916caba5fbae890eefe4f84%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **(TextReact) Predictive Chemistry Augmented with Text Retrieval**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.784.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe5c99ba5744eed11a42cc12ed1e8ddeeefe55ad3%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/thomas0809/textreact?color=yellow&style=social)](https://github.com/thomas0809/textreact)

* **MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.966v2.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F25738c43c0c4788d803981eaf5d397691aba0958%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/eltociear/MolCA?color=yellow&style=social)](https://github.com/eltociear/MolCA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://ufile.io/6vffm5bg)

* **ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.findings-emnlp.366.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe391d266b0d43475567f59efeaeabc884a48abd0%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/syr-cn/ReLM?color=yellow&style=social)](https://github.com/syr-cn/ReLM)

* **(MoleculeSTM) Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing**
  
  [![](https://img.shields.io/badge/Nature_Machine_Intelligence_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s42256-023-00759-6.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F958bb3831589246fe5b6b58cf99e3b65c58d027f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/chao1224/MoleculeSTM?color=yellow&style=social)](https://github.com/chao1224/MoleculeSTM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/chao1224/MoleculeSTM)

* **(AMAN) Adversarial Modality Alignment Network for Cross-Modal Molecule Retrieval**
  
  [![](https://img.shields.io/badge/IEEE_TAI_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://ieeexplore.ieee.org/document/10063974)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa9e916f8bbb6a08793e949eee8b5a06c74b17f36%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/NicoleBonnie/AMAN?color=yellow&style=social)](https://github.com/NicoleBonnie/AMAN)

* **MolLM: A Unified Language Model for Integrating Biomedical Text with 2D and 3D Molecular Representations**
  
  [![](https://img.shields.io/badge/BioRxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.biorxiv.org/content/10.1101/2023.11.25.568656v2.full.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2f8f4530247e9bd43483cf28e1ebf5b3791d94d2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/gersteinlab/MolLM?color=yellow&style=social)](https://github.com/gersteinlab/MolLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/17XhqdsDOxiT8PEDLHdsLPKf62PXPmbms)

* **(MolReGPT) Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.06615.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F073e4f0c3a66b7557abd053301b5104cdc582636%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/phenixace/MolReGPT?color=yellow&style=social)](https://github.com/phenixace/MolReGPT)

* **(CaR) Can Large Language Models Empower Molecular Property Prediction?**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.07443.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1212b1e44f7611d2017b246fd3d8e9c973c9d937%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ChnQ/LLM4Mol?color=yellow&style=social)](https://github.com/ChnQ/LLM4Mol)

* **MolFM: A Multimodal Molecular Foundation Model**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.09484.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2f8f4530247e9bd43483cf28e1ebf5b3791d94d2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://pan.baidu.com/share/init?surl=iAMBkuoZnNAylhopP5OgEg\&pwd=7a6b)

* **(ChatMol) Interactive Molecular Discovery with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2306.11976.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd85e98bd13a5b4c81577ab3908ff532f445ffa3a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/Ellenzzn/ChatMol?color=yellow&style=social)](https://github.com/Ellenzzn/ChatMol)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://drive.google.com/drive/folders/1I-LcBE0emj8p1W6WFCbEYajTADGN1RuC)

* **InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.16208.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2b3554a8fea6f123fc04bd3e120f2293f227e1b2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/IDEA-XL/InstructMol?color=yellow&style=social)](https://github.com/IDEA-XL/InstructMol)

* **ChemCrow: Augmenting large-language models with chemistry tools**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2304.05376.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F354dcdebf3f8b5feeed5c62090e0bc1f0c28db06%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ur-whitelab/chemcrow-public?color=yellow&style=social)](https://github.com/ur-whitelab/chemcrow-public)

* **GPT-MolBERTa: GPT Molecular Features Language Model for molecular property prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2310.03030.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F788da23e46632cca4696a4a8286e2ea9b33a1b46%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/Suryanarayanan-Balaji/GPT-MolBERTa?color=yellow&style=social)](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa)

* **nach0: Multimodal Natural and Chemical Languages Foundation Model**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.12410.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd733c2d6e08f5b59ccc0af9188f1f86d0aa7a4c5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.03907.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc77c48fe9060aa83627fc2c7f331325de0c4fdac%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/UCSD-AI4H/drugchat?color=yellow&style=social)](https://github.com/UCSD-AI4H/drugchat)

* **(Ada/Aug-T5) From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2309.05203.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc6251c3566d90caa162832eb5e5fb93a9f42a78d%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/SCIR-HI/ArtificiallyR2R?color=yellow&style=social)](https://github.com/SCIR-HI/ArtificiallyR2R)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/SCIR-HI)

* **MolTailor: Tailoring Chemical Molecular Representation to Specific Tasks via Text Prompts**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.11403.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fec193ae36aa0d1e42f5b85e62036b71ef10e3659%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/SCIR-HI/MolTailor?color=yellow&style=social)](https://github.com/SCIR-HI/MolTailor)

* **(TGM-DLM) Text-Guided Molecule Generation with Diffusion Language Model**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.13040.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4cf3bcd849baa318252e3340eceab465111390e5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/Deno-V/tgm-dlm?color=yellow&style=social)](https://github.com/Deno-V/tgm-dlm)

* **GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text**
  
  [![](https://img.shields.io/badge/Computers_in_Biology_and_Medicine_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://doi.org/10.1016/j.compbiomed.2024.108073)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2e3dcf5a5d58ac210d0d87e9f918540a8373211a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ai-hpc-research-team/git-mol?color=yellow&style=social)](https://github.com/ai-hpc-research-team/git-mol)

* **PolyNC: a natural and chemical language model for the prediction of unified polymer properties**
  
  [![](https://img.shields.io/badge/Chemical_Science_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://pubs.rsc.org/en/content/articlepdf/2024/sc/d3sc05079c)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F08e3ba7c046964aad7a875e6795c8e01d2a11e55%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/HKQiu/Unified_ML4Polymers?color=yellow&style=social)](https://github.com/HKQiu/Unified_ML4Polymers)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/hkqiu/PolyNC)

* **MolTC: Towards Molecular Relational Modeling In Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.03781.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F81448c69a0b900f3721596c635c849987eec1a4b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/MangoKiller/MolTC?color=yellow&style=social)](https://github.com/MangoKiller/MolTC)

* **T-Rex: Text-assisted Retrosynthesis Prediction**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.14637.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ffe8527541afd578326a51b885aef82da6bb32e95%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/lauyikfung/T-Rex?color=yellow&style=social)](https://github.com/lauyikfung/T-Rex)

* **LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09391.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1823b8aecd62ccfca0cb6caa8e2a1159754afc5e%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/OSU-NLP-Group/LLM4Chem?color=yellow&style=social)](https://github.com/OSU-NLP-Group/LLM4Chem)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/osunlp/LLM4Chem)

* **(Drug-to-indication) Emerging Opportunities of Using Large Language Models for Translation Between Drug Molecules and Indications**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09588.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F573dbc7d2d4f63e8c045225c03d606284290f4f8%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/PittNAIL/drug-to-indication?color=yellow&style=social)](https://github.com/PittNAIL/drug-to-indication)

* **ChemDFM: Dialogue Foundation Model for Chemistry**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.14818.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F660f5c57d459671f3f6436f116fef0bf011c1748%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **DrugAssist: A Large Language Model for Molecule Optimization**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2401.10334.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fddc1899e59a8e4fda60f5a175fef710a63abcef9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/blazerye/DrugAssist?color=yellow&style=social)](https://github.com/blazerye/DrugAssist)

* **ChemLLM: A Chemical Large Language Model**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.06852.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9f0bac228c616236c9fb8c25fbee817b1599a929%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)

* **(TEDMol) Text-guided Diffusion Model for 3D Molecule Generation**
  
  [![](https://img.shields.io/badge/OpenReview-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=FdUloEgBSE)

* **(3DToMolo) Sculpting Molecules in 3D: A Flexible Substructure Aware Framework for Text-Oriented Molecular Optimization**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.03425.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F29fdfd27b591a3af255ddedff97a6df6ca36d801%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **(ICMA) Large Language Models are In-Context Molecule Learners**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.04197.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6eb23df05166c772e4c2fbfb0113de0beabd1a43%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **Benchmarking Large Language Models for Molecule Prediction Tasks**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.05075.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2a1cd0bd878ab77742c2f120223f1a44accdd204%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/zhiqiangzhongddu/LLMaMol?color=yellow&style=social)](https://github.com/zhiqiangzhongddu/LLMaMol)

* **DRAK: Unlocking Molecular Insights with Domain-Specific Retrieval-Augmented Knowledge in LLMs**

  [![](https://img.shields.io/badge/ResearchGate_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.researchgate.net/profile/Jinzhe-Liu-2/publication/378683833_DRAK_Unlocking_Molecular_Insights_with_Domain-Specific_Retrieval-Augmented_Knowledge_in_LLMs/links/65e43d34adf2362b63683086/DRAK-Unlocking-Molecular-Insights-with-Domain-Specific-Retrieval-Augmented-Knowledge-in-LLMs.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7176288231dcc568daba50a35370d54e34ba3889%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **3M-Diffusion: Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.07179.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F004ff041d40a7912b31dbe98bbed0c2755655a5f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/huaishengzhu/3MDiffusion?color=yellow&style=social)](https://github.com/huaishengzhu/3MDiffusion)

* **(TSMMG) Instruction Multi-Constraint Molecular Generation Using a Teacher-Student Large Language Model**

  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.13244.pdf)
  [![Stars](https://img.shields.io/github/stars/HHW-zhou/TSMMG?color=yellow&style=social)](https://github.com/HHW-zhou/TSMMG)


### Text + Protein
* **OntoProtein: Protein Pretraining With Gene Ontology Embedding**
  
  [![](https://img.shields.io/badge/ICLR_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=yfe1VMYAXa4)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F10be7d45b3736cb9eac13a0c07d00c7f8e4f84b4%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/zjunlp/OntoProtein?color=yellow&style=social)](https://github.com/zjunlp/OntoProtein)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp/OntoProtein)

* **ProTranslator: Zero-Shot Protein Function Prediction Using Textual Description**
  
  [![](https://img.shields.io/badge/RECOMB_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/abs/2204.10286)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc0610b4d9f70cfc367dd302cb57a4dc7e309e3bf%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/HanwenXuTHU/ProTranslator?color=yellow&style=social)](https://github.com/HanwenXuTHU/ProTranslator)

* **ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts**
  
  [![](https://img.shields.io/badge/ICML_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://proceedings.mlr.press/v202/xu23t/xu23t.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4ea1f64c13280ef13f506eef4b3dd2395d1cf171%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/DeepGraphLearning/ProtST?color=yellow&style=social)](https://github.com/DeepGraphLearning/ProtST)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/DeepGraphLearning/ProtST)

* **InstructProtein: Aligning Human and Protein Language via Knowledge Instruction**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2310.03269.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff131b342e3aede46d24afc9b9055a94cceb0936a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **(ProteinDT) A Text-guided Protein Design Framework**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2302.04611.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd55b69a533dea69c8b2673cde8de90c6626ee789%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **ProteinChat: Towards Achieving ChatGPT-Like Functionalities on Protein 3D Structures**
  
  [![](https://img.shields.io/badge/TechRxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23120606.v1)
  [![Stars](https://img.shields.io/github/stars/UCSD-AI4H/proteinchat?color=yellow&style=social)](https://github.com/UCSD-AI4H/proteinchat)

* **Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2307.14367.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fde7e5fee8cf03bd485b1104d3e40e8ab45d76c0a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/hadi-abdine/Prot2Text?color=yellow&style=social)](https://github.com/hadi-abdine/Prot2Text)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/hadi-abdine/Prot2Text)
  
* **ProtChatGPT: Towards Understanding Proteins with Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.09649.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F61f4b39325cf28026f3491ec78d938c78bb50dda%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **ProtAgents: Protein discovery via large language model multi-agent collaborations combining physics and machine learning**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.04268.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F797597040167622a126c3206bbf94459238e2c19%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/lamm-mit/ProtAgents?color=yellow&style=social)](https://github.com/lamm-mit/ProtAgents)

* **ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.16445v1.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1296d51001f8a73c0a3356f78b136a691928985c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/Lyu6PosHao/ProLLaMA?color=yellow&style=social)](https://github.com/Lyu6PosHao/ProLLaMA)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/GreatCaptainNemo/ProLLaMA)

* **ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.07920.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F631066c55a852fac7b9c9129e166550a6310fa3c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ProtLLM/ProtLLM?color=yellow&style=social)](https://github.com/ProtLLM/ProtLLM)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/datasets/ProtLLM/ProtLLM)


### More Modalities

* **Galactica: A Large Language Model for Science**
  
  [![](https://img.shields.io/badge/Arxiv_2022-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://galactica.org/static/paper.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7d645a3fd276918374fd9483fd675c28e46506d1%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/paperswithcode/galai?color=yellow&style=social)](https://github.com/paperswithcode/galai)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/models?other=galactica)

* **BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations**
  
  [![](https://img.shields.io/badge/EMNLP_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://aclanthology.org/2023.emnlp-main.70.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc3382fd533b9dd7f8ed7ba7766159079bc1d3935%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/QizhiPei/BioT5?color=yellow&style=social)](https://github.com/QizhiPei/BioT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/QizhiPei/biot5-base)

* **DARWIN Series: Domain Specific Large Language Models for Natural Science**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.13565.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4c6fb350e7769cb730a15c62927b6e9b563d0157%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/MasterAI-EAM/Darwin?color=yellow&style=social)](https://github.com/MasterAI-EAM/Darwin)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/MasterAI-EAM/Darwin)

* **BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2308.09442.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe3fd89a7f6b28973cfc68bfc51caebd8fb93f0bc%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?color=yellow&style=social)](https://github.com/PharMolix/OpenBioMed)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://pan.baidu.com/share/init?surl=iAMBkuoZnNAylhopP5OgEg\&pwd=7a6b)

* **(StructChem) Structured Chemistry Reasoning with Large Language Models**
  
  [![](https://img.shields.io/badge/Arxiv_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2311.09656.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe56aa728aaa32c087c8f7bc56a7eb225675dd8ae%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/ozyyshr/StructChem?color=yellow&style=social)](https://github.com/ozyyshr/StructChem?tab=readme-ov-file)

* **(BioTranslator) Multilingual translation for zero-shot biomedical classification using BioTranslator**
  
  [![](https://img.shields.io/badge/Nature_Communications_2023-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://www.nature.com/articles/s41467-023-36476-2.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fce6f2d68b1a4029ff4a838fcf12d5ad1d47f0e68%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/HanwenXuTHU/BioTranslatorProject?color=yellow&style=social)](https://github.com/HanwenXuTHU/BioTranslatorProject)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://figshare.com/articles/dataset/Protein_Pathway_data_tar/20120447)

* **Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=Tlsdsb6l9n)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F984b36e80c7f46c1102a8904cb2236b58815931b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/zjunlp/Mol-Instructions?color=yellow&style=social)](https://github.com/zjunlp/Mol-Instructions)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp)

* **(ChatDrug) ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=yRrPfKyJQ2)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9e8af0791e8c87452c8cff25dab5448a29c218d4%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/chao1224/ChatDrug?color=yellow&style=social)](https://github.com/chao1224/ChatDrug)

* **BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs**
  
  [![](https://img.shields.io/badge/ICLR_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://openreview.net/pdf?id=jJCeMiwHdH)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F16e0b8c878c75bb57ffb62c08ebf23b51ac10b99%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/RyanWangZf/BioBridge?color=yellow&style=social)](https://github.com/RyanWangZf/BioBridge)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/RyanWangZf/BioBridge/tree/main/checkpoints)

* **(KEDD) Towards Unified AI Drug Discovery with Multiple Knowledge Modalities**
  
  [![](https://img.shields.io/badge/AAAI_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2305.01523.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4810273202207675ef9b3efcbda95b271709ea71%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **ChatCell: Facilitating Single-Cell Analysis with Natural Language**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.08303v2.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6a3c0c897870c4f1897d2af0b88981fa2764359b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/zjunlp/ChatCell?color=yellow&style=social)](https://github.com/zjunlp/ChatCell)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://huggingface.co/zjunlp)

* **BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.17810.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff740a2474b52675287166a003bd1313f8aabcd68%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/QizhiPei/BioT5?color=yellow&style=social)](https://github.com/QizhiPei/BioT5)
  [![Model](https://img.shields.io/badge/Model-5291C8?style=flat&logo=themodelsresource&labelColor=555555)](https://github.com/QizhiPei/BioT5)

* **MolBind: Multimodal Alignment of Language, Molecules, and Proteins**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.08167.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe9ef2567e4c8f8b47f1110a2a3959c20bd218c0d%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/tengxiao1/MolBind?color=yellow&style=social)](https://github.com/tengxiao1/MolBind)

* **Uni-SMART: Universal Science Multimodal Analysis and Research Transformer**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2403.10301.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7130fe366c9bf8fe64e5d6dcabb292431f69fba9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)

* **Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.05140.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb900b97d7657bdac9be8badb948a18e5eacefb9c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)
  [![Stars](https://img.shields.io/github/stars/sjunhongshen/Tag-LLM?color=yellow&style=social)](https://github.com/sjunhongshen/Tag-LLM)

* **An Evaluation of Large Language Models in Bioinformatics Research**
  
  [![](https://img.shields.io/badge/Arxiv_2024-5291C8?style=flat&logo=Read.cv&labelColor=555555)](https://arxiv.org/pdf/2402.13714.pdf)
  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb26f66c9bb59811c7d67cdab91a0d65d15364836%3Ffields%3DcitationCount&query=%24.citationCount&label=citation&style=social&labelColor=555555&color=ED8936)


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
| MoleculeQA           | Fine-tuning       | Text, Molecule          | [https://github.com/IDEA-XL/MoleculeQA](https://github.com/IDEA-XL/MoleculeQA) |
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
| DARWIN               | Instruction       | Text, Molecule, etc     | [https://github.com/MasterAI-EAM/Darwin/tree/main/dataset](https://github.com/MasterAI-EAM/Darwin/tree/main/dataset) |
| StructChem           | Instruction       | Text, Molecule, etc     | [https://github.com/ozyyshr/StructChem](https://github.com/ozyyshr/StructChem) |
| SciAssess            | Instruction       | Text, Molecule, etc     | [https://sci-assess.github.io](https://sci-assess.github.io/), [https://github.com/sci-assess/SciAssess](https://github.com/sci-assess/SciAssess) |
| InstructProtein      | Instruction       | Text, Protein           | - |
| Open Protein Instructions | Instruction | Text, Protein          | [https://github.com/baaihealth/opi](https://github.com/baaihealth/opi) |
| Mol-Instructions     | Instruction       | Text, Molecule, Protein | [https://huggingface.co/datasets/zjunlp/Mol-Instructions](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |
| CheF                 | -                 | Text, Molecule          | [https://github.com/kosonocky/CheF](https://github.com/kosonocky/CheF) |
| IUPAC Gold Book      | -                 | Text, Molecule          | [https://goldbook.iupac.org](https://goldbook.iupac.org/) |
| ChemNLP              | -                 | Text, Molecule, etc     | [https://github.com/OpenBioML/chemnlp](https://github.com/OpenBioML/chemnlp) |
| ChemFOnt             | -                 | Text, Molecule, Protein, etc | [https://www.chemfont.ca](https://www.chemfont.ca)|



## Related Resources
### Related Surveys & Evaluations
* Bridging Text and Molecule: A Survey on Multimodal Frameworks for Molecule [Arxiv 2403](https://arxiv.org/abs/2403.13830)
* From Words to Molecules: A Survey of Large Language Models in Chemistry [Arxiv 2402](https://arxiv.org/abs/2402.01439)
* Scientific Language Modeling: A Quantitative Review of Large Language Models in Molecular Science [Arxiv 2402](https://arxiv.org/abs/2402.04119)
* Scientific Large Language Models: A Survey on Biological & Chemical Domains [Arxiv 2401](https://arxiv.org/abs/2401.14656)
* The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 [Arxiv 2311](https://arxiv.org/abs/2311.07361)
* Transformers and Large Language Models for Chemistry and Drug Discovery [Arxiv 2310](https://arxiv.org/abs/2310.06083)
* Language models in molecular discovery [Arxiv 2309](https://arxiv.org/abs/2309.16235)
* What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks [NeurIPS 2309](https://openreview.net/pdf?id=1ngbR3SZHW)
* Do Large Language Models Understand Chemistry? A Conversation with ChatGPT [JCIM 2303](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00285)
* A Systematic Survey of Chemical Pre-trained Models [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0760.pdf)


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
This repository is contributed and updated by [QizhiPei](https://qizhipei.github.io) and [Lijun Wu](https://apeterswu.github.io). If you have questions, don't hesitate to open an issue or ask me via <qizhipei@ruc.edu.cn> or Lijun Wu via <lijuwu@microsoft.com>. We are happy to hear from you!

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
