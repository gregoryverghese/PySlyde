---
title: 'PySlyde: A Lightweight, Open-Source Toolkit for Pathology Preprocessing'
tags:
  - Python
  - computational pathology
  - whole-slide images
  - digital pathology
  - image preprocessing
  - foundation models
authors:
  - name: Gregory Verghese
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2"
  - name: Anthony Baptista
    equal-contrib: true
    affiliation: "2, 3"
  - name: Chima Eke
    equal-contrib: true
    affiliation: 2
  - name: Holly Rafique
    equal-contrib: true
    affiliation: 2
  - name: Elizabeth Ing-Simmons
    equal-contrib: true
    affiliation: "1, 4"
  - name: Enrico Parisini
    affiliation: "1, 2"
  - name: Mengyuan Li
    affiliation: "1, 2"
  - name: Fathima Mohamed
    affiliation: 2
  - name: Ananya Bhalla
    affiliation: "2, 5"
  - name: Lucy Ryan
    affiliation: 2
  - name: Michael Pitcher
    affiliation: "1, 2"
  - name: Concetta Piazzese
    affiliation: "1, 6"
  - name: James Graham
    affiliation: "1, 4"
  - name: Dinis Pedro Calado
    affiliation: 5
  - name: Christopher R.S. Banerji
    affiliation: "2, 3"
  - name: Anita Grigoriadis
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: PharosAI, London, UK
    index: 1
  - name: Cancer Bioinformatics, School of Cancer and Pharmaceutical Sciences, Faculty of Life Sciences and Medicine, King's College London, London, UK
    index: 2
  - name: The Alan Turing Institute, The British Library, London, UK
    index: 3
  - name: eResearch, King's College London, London, UK
    index: 4
  - name: The Francis Crick Institute, London, UK
    index: 5
  - name: Barts Life Sciences, Barts Health NHS Trust, London, UK
    index: 6
date: 9 March 2026
bibliography: paper.bib
---

# Summary

Artificial intelligence (AI) represents a transformative opportunity for precision medicine, with the potential to radically improve patient outcomes [@Verghese2023ComputationalPI; @Song2023ArtificialIF]. Advances in computational pathology are enabling the analysis of complex histopathological data at unprecedented scale and depth. By combining modern computer vision, machine learning, and pathology, researchers can predict quantitative, molecular [@Dey2025GeneratingCG] and digital biomarkers, detect morphological and tissue patterns [@Chen2026NormalBT], and develop predictive models for diagnostics, prognosis [@Verghese2023MultiscaleDL] and treatment response to complement human expertise [@vanderLaak2021DeepLI].

Whole slide images (WSIs), resulting from the digitisation of histopathology slides, offer valuable spatial and morphological insights that can deepen our understanding of cancer biology. However, their size (gigapixel-scale), complexity, and variability introduce significant computational and standardisation challenges for downstream tasks [@McGenity2023ArtificialII]. Preprocessing WSIs, including tissue detection, tessellation (splitting into smaller tiled images), stain normalisation, and annotation parsing, are essential steps before any downstream AI analysis can occur [@Pocock2022TIAToolboxAA]. Building robust and reproducible pipelines for WSI preprocessing is fundamental to computational pathology research. However, existing workflows often rely on ad hoc proprietary scripts, fragmented tools, and inconsistent formats, making standardisation and reproducibility difficult.

PySlyde is a lightweight, open-source Python toolkit built on top of OpenSlide [@GOODE201327], designed to address this gap and provide an intuitive approach to quickly preprocess WSIs. It provides a simple API to perform WSI loading, annotation handling, tissue detection, tile generation, and feature extraction with support for the latest histopathology foundation models. PySlyde aims to lower the barrier to entry for pathology researchers, standardise preprocessing workflows, and accelerate the development of AI-ready datasets for computational pathology research, enabling researchers to focus on model development and other downstream tasks.

![Schematic overview of the WSI processing workflow supported by PySlyde, illustrating the core functionalities across the main classes and modules `Slide`, `WSIParser`, `FeatureGenerator`, and `IO`. Top row, from left to right: (i) WSI loading using compatible formats through OpenSlide; (ii) annotation mask generation with support for QuPath or ImageJ formats; and (iii) preprocessing steps such as tissue detection. Bottom row, from left to right: (iv) tessellation of WSIs into tissue tiles using the `WSIParser` class; (v) feature extraction from tiles with support for pretrained encoders such as Virchow2, H-optimus, Gigapath, or UNI; and (vi) structured storage of tile embeddings and metadata using RocksDB, NumPy, or LMDB.\label{fig:pipeline}](fig1.png){ width=100% }

# Statement of Need

WSI preprocessing is a foundational but time-consuming step in computational pathology. Common tasks, such as annotation parsing, binary and annotation mask generation, stain normalisation, and tessellation, often involve custom scripts, multiple packages, and ad hoc pipelines built for a single dataset or project. This fragmentation hinders reproducibility, increases technical debt, and slows the development and translation of AI methods into clinical workflows. The lack of standardised preprocessing practices thus remains one of the key barriers preventing the seamless integration of AI into diagnostic pathology [@bilal2025foundationmodelscomputationalpathology].

While powerful frameworks exist for model training and inference, few provide a simple, consistent interface for preparing WSIs at scale. PySlyde directly addresses this unmet need by providing a modular, well-documented, and extensible preprocessing toolkit that integrates seamlessly with existing ecosystems, including OpenSlide, NumPy, PyTorch, and modern foundation models, enabling researchers to build reproducible, scalable pipelines with minimal code.

# State of the Field

Several open-source tools support whole-slide image analysis in computational pathology. At the foundational level, OpenSlide [@GOODE201327] provides a widely adopted C library and Python bindings for programmatic access to WSI formats. It enables efficient tile-based image loading, thumbnail generation, and extraction of slide metadata across multiple vendor-specific file formats. While it provides essential low-level functionality for accessing WSI data, higher-level workflows such as annotation management, spatial analysis, and feature extraction must typically be implemented by downstream libraries.

Building upon such low-level infrastructure, frameworks such as TIAToolbox [@Pocock2022TIAToolboxAA] provide comprehensive ecosystems for tissue analytics, model development, and deployment. These frameworks offer extensive end-to-end functionality including model training, inference, and visualisation. In contrast, PySlyde is designed specifically as a lightweight, modular preprocessing layer for WSI analysis, focusing on efficient data preparation, annotation handling, and feature embedding generation for downstream machine learning workflows.

PySlyde provides native support for annotations generated by widely adopted WSI analysis tools such as QuPath [@Bankhead2017QuPathOS] and ImageJ [@Schneider2012NIHIT]. Annotation files (e.g., QuPath JSON exports) are parsed into structured spatial representations that preserve region geometry and associated metadata. These annotations can be converted into binary or multi-class masks for pixel-level preprocessing, or represented as GeoDataFrame polygon objects that retain slide-level spatial coordinates. Representing annotations in this format enables interoperability with geospatial libraries such as GeoPandas and Shapely, facilitating spatial analysis workflows that extend beyond traditional image processing pipelines. Annotations are exported to interoperable formats such as GeoJSON, enabling integration with external visualisation tools and GIS-based analytical workflows.

PySlyde further differentiates itself through its native support for histopathology foundation models and structured feature extraction. Embedding extraction from modern histopathology foundation models—Virchow2 [@Zimmermann2024Virchow2SS], H-optimus [@hoptimus2024], UNI [@Chen2024TowardsAG], and cTransPath [@Wang2022TransformerbasedUC]—is treated as a first-class preprocessing component rather than an auxiliary utility. Extracted embeddings and associated metadata are exported to scalable key–value storage backends such as LMDB and RocksDB, enabling efficient indexed retrieval for large-scale machine learning and representation learning workflows.

PySlyde addresses a specific gap in the current ecosystem: providing a minimal, modular preprocessing backbone that integrates WSI annotation handling, geospatial representations, and foundation model feature extraction within a single lightweight framework. This design enables researchers to incorporate modern foundation model embeddings and spatially aware annotation data into scalable machine learning pipelines without adopting a full end-to-end pathology framework.

# Software Design

PySlyde is designed as a lightweight and modular preprocessing framework for WSI analysis. The architecture emphasises separation of functionality between slide representation, spatial parsing, feature extraction, and data persistence. This design allows individual components of the preprocessing pipeline to be used independently while maintaining a coherent workflow for large-scale computational pathology experiments. Rather than providing a tightly coupled end-to-end pipeline, PySlyde prioritises modularity, enabling researchers to integrate specific preprocessing stages into existing machine learning workflows.

At the core of the architecture is the `Slide` abstraction, which represents a single WSI together with its associated spatial metadata, annotations, and derived data. The `Slide` class inherits directly from the OpenSlide Python interface, allowing PySlyde to extend low-level slide access with higher-level functionality for annotation management, spatial masking, and tissue-aware preprocessing. This approach avoids re-implementing foundational image I/O functionality while preserving compatibility with the broad range of vendor-specific formats supported by OpenSlide.

The remaining architecture is organised around three complementary components. The `WSIParser` class handles spatial partitioning and tile extraction, separating tiling logic from slide representation so that large gigapixel images are processed efficiently under different region-selection and magnification strategies. The `FeatureGenerator` class encapsulates feature extraction workflows for modern histopathology foundation models, allowing embedding generation to remain decoupled from upstream slide parsing and tiling operations. The `IO` module manages export and persistence of tiles, embeddings, and metadata through both file-based outputs and scalable key-value stores such as Lightning Memory Mapped Database (LMDB) and RocksDB, supporting reproducible downstream analysis.

Several architectural trade-offs guided this design. First, PySlyde prioritises modularity over a monolithic pipeline structure. While tightly integrated frameworks simplify end-to-end workflows, they tend to constrain researchers to predefined processing stages. PySlyde instead provides composable abstractions that support experimentation with different tiling strategies, preprocessing methods, storage backends, and feature extractors. Second, PySlyde deliberately builds on established libraries such as OpenSlide and HistoQC rather than re-implementing low-level functionality. This reduces maintenance overhead and improves interoperability with existing pathology infrastructure, while allowing development effort to focus on annotation-aware preprocessing and foundation model integration. Together, these design choices support scalable and reproducible preprocessing workflows for computational pathology research, particularly where large WSI collections must be transformed into structured datasets for modern machine learning and representation learning applications.

# Research Impact Statement

PySlyde has been used in multiple ongoing computational pathology projects within the Cancer Bioinformatics Team at King's College London, including multimodal outcome prediction studies in breast, head and neck, and pan-cancer cohorts, with several manuscripts currently under review or in preparation. It is also used within the PharosAI Research Ventures Catalyst programme for large-scale histopathology data curation across collaborating institutions including Guy's and St Thomas' NHS Foundation Trust, Queen Mary University of London, Barts Health NHS Trust, and King's College London. These applications, alongside a poster at the *ESMO AI Congress, Berlin 2025*, demonstrate its immediate research utility and support expectations for wider adoption within computational pathology workflows.

# Use of Generative AI

No generative AI tools were used in the development of the original PySlyde codebase. More recently, AI-assisted development tools (e.g., Cursor v2.4.31) have been used to support limited refactoring (e.g., type annotations) and code maintenance tasks. Recent AI-assisted changes were reviewed through standard code review and CI/CD validation processes on GitHub to ensure correctness and reliability. Generative AI tools were also used to support the writing, reviewing, and editing of this manuscript, with all final content verified and edited by all the authors.

# Acknowledgements

We thank members of the King's College London Cancer Bioinformatics Team, the PharosAI Research Ventures Catalyst Programme, and collaborators at The Alan Turing Institute, The Francis Crick Institute, and Barts Life Sciences for feedback, testing, and support. Holly Rafique, Pre-Doctoral Fellow NIHR303406, is funded by the NIHR. Anita Grigoriadis, Gregory Verghese, Mengyuan Li, Elizabeth Ing-Simmons, and Enrico Parisini are funded by the UK government, Research Ventures Catalyst Programme, Department of Science Innovation and Technology, and the Guy's Cancer Charity under PharosAI for this research project. Anthony Baptista, Chima Eke, and Anita Grigoriadis acknowledge support from the CRUK City of London Centre Award [CTRQQR-2021/100004]. Christopher R.S. Banerji is supported by a King's College London AI+ Fellowship. Dinis Pedro Calado acknowledges support from The Francis Crick Institute, which receives core funding from Cancer Research UK (CC2078), the UK Medical Research Council (CC2078), and the Wellcome Trust (CC2078). Dinis Pedro Calado and Anita Grigoriadis acknowledge support from the UK Medical Research Council (MR/W025221/1). Lucy Ryan is funded by the Pathological Society of Great Britain and Ireland and the Jean Shanks Foundation (JSPS CPF 1023 05). The views expressed in this publication are those of the author(s) and not necessarily those of the NIHR.

# References
