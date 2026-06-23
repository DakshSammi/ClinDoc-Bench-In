# Server 2 Import Blocked

Automated Server 2 import was attempted from Server 1 with:

```bash
ssh -p 7722 nlp@10.10.110.37 "hostname && ls /mnt/mnfas_Varahi/Daksh | head"
```

The host key was accepted, but non-interactive authentication failed with `Permission denied (publickey,password)`. The clean repository therefore includes Server 1 assets only plus this import request.

Copy these files manually from Server 2 when available:

- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_final_paper_table.md`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_ppt_aligned_metrics.csv`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_raw_ocr_ppt_table.csv`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_internal_qwen3_merged_ppt_metrics.csv`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_internal_qwen3_merged_final_summary.md`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_model_coverage_matrix.csv`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1b_server2_paper_results_snapshot.md`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1c_server2_semantic_summary.md`
- `/mnt/mnfas_Varahi/Daksh/reports/stage1c_server2_semantic_vs_raw_interpretation.md`

Optional clean code imports, excluding raw data and outputs:

- `/mnt/mnfas_Varahi/Daksh/scripts/`
- `/mnt/mnfas_Varahi/Daksh/src/`
- `/mnt/mnfas_Varahi/Daksh/configs/`
- `/mnt/mnfas_Varahi/Daksh/prompts/`
- `/mnt/mnfas_Varahi/Daksh/docs/`
