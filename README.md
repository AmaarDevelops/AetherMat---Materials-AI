# AetherMat

**AI-driven materials discovery for advanced thermal management**

AetherMat is an AI-native materials company focused on discovering and optimizing materials for extreme thermal environments. We combine machine learning, materials informatics, and physics-based screening to generate high-value, protectable material solutions.

Current focus is dual-track:

- **Track A (Primary)**: Next-generation thermal management materials for semiconductor packaging and high-power electronics
- **Track B (Secondary)**: Thermal and extreme-environment materials relevant to aerospace and space applications

---

## Vision

Build a defensible materials technology platform that:

1. Discovers or enables materials with genuine performance advantages
2. Generates protectable intellectual property
3. Supports near-term commercial relevance (semiconductor packaging) while building longer-term position in higher-barrier markets (aerospace / space)

We do **not** aim to rediscover known filler loadings or incremental variants of AlN, BN, or BeO. The system is designed to favor novelty, synthesizability, and clear differentiation.

---

## Dual-Track Strategy

### Track A – Semiconductor Packaging & High-Power Electronics (Primary)

**Goal**: Create differentiated thermal management solutions for advanced packaging, power modules, and high-temperature electronics.

Focus areas:
- Interfacial and surface chemistry innovations that improve effective thermal conductivity or reliability
- Non-obvious hybrid / hierarchical filler architectures
- Materials and and processes for next-generation power electronics and high-temperature packages
- Solutions that offer clear advantages in thermal performance, processability, CTE management, or reliability

Success criteria:
- Measurable advantage over current commercial baselines
- Pathway to protectable IP (composition, process, architecture, or application-specific formulation)
- Feasibility for development partnerships or paid projects

### Track B – Aerospace & Space Materials (Secondary)

**Goal**: Build early capability and IP position in high-value extreme-environment materials.

Initial focus areas:
- High-temperature thermal management materials
- Lightweight thermal-structural materials
- Materials with combined thermal and radiation-relevant properties
- Concepts relevant to thermal protection and extreme-temperature operation

Success criteria:
- Novel or significantly improved material concepts
- Compatibility with long qualification cycles
- Potential for future licensing or joint development with aerospace partners

Track B currently receives lower resource allocation but is deliberately maintained so the platform can expand into higher-barrier markets.

---

## Technical Approach

The platform combines:

- Materials Project / AFLOW-derived data and custom datasets
- Composition and structure-aware machine learning models (thermal conductivity, stability, and related properties)
- Physics-based filters (thermodynamic stability, charge balance, complexity, synthesizability proxies)
- Multi-objective ranking that includes performance, novelty, and practical constraints
- Iterative human-in-the-loop refinement

Key principles:
- Novelty and differentiation are first-class objectives
- Known stable phases and trivial doped variants are actively filtered
- Models are calibrated against real commercial and literature baselines
- Every candidate must survive stability and practicality filters before being considered interesting

---

## Repository Structure (Current)

```
AotherMat/
├── archive/                  # Previous phases (baseline & negative results)
│   └── phase1-7_baseline/
├── shared/                   # Reusable models, featurization, utilities
│   ├── models/
│   ├── featurization/
│   └── screening/
├── track_a_packaging/        # Primary – semiconductor thermal management
├── track_b_space/            # Secondary – aerospace / space materials
├── data/                     # Datasets and caches
└── README.md
```

Note: The repository is being restructured from the earlier single-track exploratory work (Phases 1–7). Useful models and tools are retained; low-novelty generative strategies are being replaced.

---

## Current Status

**Completed (baseline)**
- End-to-end pipeline for composition-based generation and screening
- Direct thermal conductivity model trained on AFLOW-AGL data
- Stability screening against Materials Project phase diagrams
- Composite effective-medium screening calibrated against commercial TIM / gap-pad performance
- Clear understanding of failure modes (collapse to known phases and commodity filler systems)

**In progress**
- Restructuring into dual-track architecture
- Redefining success criteria and novelty filters
- Redesigning Track A search to target higher-IP opportunities
- Standing up initial Track B capability

**Not goals**
- Publishing incremental doped versions of AlN, BN, or BeO
- Optimizing commodity hBN/AlN loadings in standard epoxies without clear differentiation

---

## Intellectual Property Philosophy

We prioritize directions where a genuine technical advantage can be protected. This includes:

- New compositions or structure types
- Novel interfacial / surface chemistries
- Non-obvious architectures or hybrid systems
- Process innovations that enable previously inaccessible performance

Pure optimization of well-known systems is treated as baseline work, not core IP.

---

## Roadmap (Near Term)

1. Finalize dual-track structure and success criteria
2. Rebuild Track A search with explicit novelty and differentiation filters
3. Establish lightweight initial Track B models and target property spaces
4. Generate and rigorously filter first new candidate sets under the updated criteria
5. Identify opportunities for early collaboration or paid development work in packaging while advancing longer-term space-relevant capabilities

---

## Contact & Collaboration

AotherMat is building toward partnerships with semiconductor packaging, power electronics, and aerospace organizations. We are open to:

- Collaborative development projects
- Joint evaluation of promising candidates
- Strategic discussions on thermal materials challenges

---

*This project was previously an exploratory multi-phase materials discovery effort. It has been refocused as the technical foundation of an AI materials company with clear dual-track priorities and an emphasis on protectable innovation.*
