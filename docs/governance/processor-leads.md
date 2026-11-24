# Processor leads

Each `bps-*` module has a **scientific module expert** (when applicable) and a
**technical representative**. GitHub routes reviews through `CODEOWNERS` using
these assignments. If your pull request touches a processor directory, expect
review from the contacts below.

---

## Ownership table

| Processor                 | Chain          | Scientific rep.   | Sci. affiliation | Technical rep. | Tech. affiliation | Notes                          |
| ------------------------- | -------------- | ----------------- | ---------------- | -------------- | ----------------- | ------------------------------ |
| bps-common                | Shared library | —                 | —                | R. Piantanida  | Aresys            | Shared utilities               |
| bps-task-tables           | Config         | —                 | —                | R. Piantanida  | Aresys            | Task table definitions         |
| bps-transcoder            | Infra          | —                 | —                | R. Piantanida  | Aresys            | Data transcoding               |
| bps-dockerfiles           | Infra          | —                 | —                | R. Piantanida  | Aresys            | Container definitions          |
| bps-l1_binaries           | L1             | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Pre-built binaries             |
| bps-stack_binaries        | Stack          | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Pre-built binaries             |
| bps-l1_pre_processor      | L1             | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Pre-processing                 |
| bps-l1_framing_processor  | L1             | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Frame definition               |
| bps-l1_core_processor     | L1             | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Core focusing and calibration  |
| bps-l1_processor          | L1             | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | End-to-end L1 orchestration    |
| bps-stack_pre_processor   | Stack          | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Stack pre-processing           |
| bps-stack_coreg_processor | Stack          | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Co-registration                |
| bps-stack_cal_processor   | Stack          | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Stack calibration              |
| bps-stack_processor       | Stack          | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | End-to-end stack orchestration |
| bps-l2a_processor         | L2a            | S. Tebaldini      | PoliMi           | R. Piantanida  | Aresys            | Backscatter retrieval          |
| bps-l2b_agb_processor     | L2b AGB        | L. Ulander        | Chalmers         | R. Piantanida  | Aresys            | Above-ground biomass           |
| bps-l2b_fh_processor      | L2b FH         | K. Papathanassiou | DLR              | R. Piantanida  | Aresys            | Forest canopy height           |
| bps-l2b_fd_processor      | L2b FD         | L. Ferro-Famil    | CESBIO           | R. Piantanida  | Aresys            | Forest disturbance             |

---

**Previous:** [Repository stewards](repository-stewards.md)
