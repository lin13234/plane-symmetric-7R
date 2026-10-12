# Plane-Symmetric 7R Mechanism: Exact Configuration-Space Topology Certificates and Locomotion Artifacts

This repository contains the exact computer-algebra verification scripts, MuJoCo mechanical models, policy checkpoints, and paired-deployment evaluation datasets accompanying the manuscript:

> **Capability-Oriented Reconstruction and Multicycle Branch Connectivity of a Plane-Symmetric 7R Mechanism with Branch-Derived Locomotion**

---

## 1. Quick Start: Reproducing the Exact Topology Certificates

All statements marked `[EXACT PROOF]` in Section 3 of the main text and Sections S2–S13 of the Supplementary Material use exact rational and algebraic-number arithmetic in SymPy (`sympy==1.14.0`, tested on Python `3.13.9`). No floating-point threshold enters any zero or rank test.

To run all ten verification scripts in sequence (~11 seconds on a desktop CPU):

```bash
pip install -r requirements.txt
cd topology_proof
python run_all_certificates.py
```

### Scripts executed by `run_all_certificates.py`

| Script | Verified statement |
|---|---|
| `certify_coverage_identities.py` | Global coverage identities (Supplement Sections S5, S6) |
| `certify_quartic_discriminant.py` | Reduced spatial quartic, Cramer identity, and discriminant from MDH parameters |
| `certify_s0_rotation_components.py` | Exact rotation components for $\sin q_3 = 0$ and general $q_7$ |
| `certify_folded_planar_cycles.py` | Closure and Jacobian rank on the four detached folded planar circles |
| `certify_conventional_planar_rank.py` | Full-loop Jacobian rank census on the four conventional planar circles |
| `certify_spatial_regular_rank.py` | Full-loop Jacobian rank census on the regular spatial arcs |
| `certify_exceptional_spatial_points.py` | Closure and rank at the exceptional atlas points (Supplement Section S7) |
| `certify_spatial_sheet_gluing.py` | Leading coefficients and Sturm root counts for spatial sheet gluing |
| `certify_gateway_rank.py` | Exact closure and rank 5 at the eight bifurcation interfaces $G_1,\dots,G_4,H_1,\dots,H_4$ |
| `certify_interface_transversality.py` | Block structure, two-dimensional kernel, and transversality at the interfaces (Lemma 1) |

In addition, `reduction_recovery/recovered_identity_checks.py` verifies the 15 polynomial reduction identities bridging the full 7R loop equations to the spatial quartic:

```bash
cd reduction_recovery
python recovered_identity_checks.py
```

---

## 2. Repository Structure

```text
plane-symmetric-7R/
├── topology_proof/                # Exact SymPy certificates for c-space topology (b0 = 6, b1 = 14)
│   ├── run_all_certificates.py    # One-command runner for all 10 certificate scripts
│   ├── exact_model.py             # Exact symbolic MDH kinematics over Q
│   └── *.py / *.json              # Individual verification scripts and output certificates
├── reduction_recovery/            # Algebraic reduction identity checks
├── model/                         # MuJoCo XML definitions and CAD link meshes (.STL)
│   ├── zhengti_joint2_v2.xml
│   ├── zhengti_stage2_mesh_collision.xml
│   └── meshes/                    # 8 STL link meshes
├── stage1/                        # Stage I (fixed-root, contact-free) evaluation artifacts
│   ├── actuation_interface_audit/ # Actuated-coordinate projection rank & singular values at interfaces
│   ├── audit_full_ideal_set.py    # Distance evaluation over all 12 analytical cycles (Eq. 65)
│   ├── regular_initial_controls/  # 12 regular-initial-state Stage I controls (A3/A6 x BS1/BS3 x seeds 7/42/2026)
│   └── *.csv / *.json             # State-sequence summaries and transition-fidelity audits (Table 4, Fig. 8)
└── stage2/                        # Stage II (free-root ground locomotion) code, policies, and results
    ├── fourier_references/        # Analytical branch-derived nominal Fourier references (B_P, B_S1)
    ├── checkpoints/               # Trained PPO residual policy checkpoints (.pt, seeds 42, 43, 44)
    │   ├── S01/                   # Planar locomotion primitive
    │   ├── S02/                   # First spatial locomotion primitive
    │   ├── S03_retrained/         # Second spatial primitive (retrained under full collision model)
    │   └── S03_initial/           # Initial S03 checkpoints retained for provenance comparison
    ├── paired_replay_results/     # 120 paired perturbation deployments (input_gate & strict_gate; Table 8, Fig. 10)
    ├── cycle_diagnostics/         # 1540 measured complete intervals of all 120 deployments (Supplement figure)
    ├── embodiment_interference_audit/ # Link-mesh collision audit along analytical cycles (Fig. 7)
    ├── scripts/                   # Paired replay evaluation and training entry scripts
    ├── tasks_s01/                 # Task environment definitions for S01
    └── tasks_s02_s03/             # Task environment definitions for S02 and S03
```

---

## 3. Mapping to Manuscript Figures and Tables

- **Section 3 & Table 3 (Configuration-space census, $b_0=6$, $b_1=14$)**: `topology_proof/run_all_certificates.py` and `topology_proof/all_certificates_summary.json`.
- **Section 4.1 & Figure 7 (Embodiment collision audit)**: `stage2/embodiment_interference_audit/interference_audit.py` and `interference_audit.json`.
- **Section 4.2, Table 4 & Figure 8 (Stage I state sequences & transition fidelity, Eq. 65)**: `stage1/branch_sequence_by_replay.csv`, `stage1/transition_path_audit_v2.csv`, and `stage1/actuation_interface_audit/actuation_interface_audit.json`.
- **Section 4.2 (Regular-initial-state controls) & Supplement Table "Twelve complete regular-state evaluations"**: `stage1/regular_initial_controls/stage1_regular_complete_summary.csv` and `stage1_review.json` (0/12 strictly certified switches).
- **Supplement "Measured-cycle diagnostics for the 120 paired deployments"**: `stage2/cycle_diagnostics/cycle_source_data.csv` (1540 intervals, 72 negative) and `original_and_diagnostic_cycle_comparison.csv`.
- **Section 4.3, Table 8 & Figure 10 (Stage II paired zero-residual comparisons)**: `stage2/paired_replay_results/input_gate/paired_analysis.json` and `derived_output_pair_differences.csv` (primary 10-trial input-gate protocol), and `stage2/paired_replay_results/strict_gate/paired_analysis.json` (strict-gate sensitivity check).

---

## 4. Version history

- **v1.1.0**: added the twelve regular-initial-state Stage I controls and the measured-cycle diagnostics of the 120 paired deployments, plus independent statistics review files.
- **v1.0.0**: initial release.

## 5. License

This code and data package is released under the [MIT License](LICENSE).
