| model | track | origin_server | coverage | publication_status | reason_if_excluded | statistics_used |
| --- | --- | --- | --- | --- | --- | --- |
| glm_ocr | raw_ocr | server1 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| doctr | raw_ocr | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| easyocr | raw_ocr | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| trocr | raw_ocr | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| surya | raw_ocr | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| docling | raw_ocr | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| qwen3_vl_raw_ocr | raw_ocr | server1 | 125/125 | PRIMARY TABLE |  | wilcoxon;friedman;bootstrap_ci;holm |
| qwen3_vl | direct_vlm | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| google_gemini_2_5_flash | direct_vlm | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| hf_qwen25_vl_72b | direct_vlm | server2 | 76/125 | APPENDIX | partial imported lane | descriptive_only |
| qwen3_27b | direct_vlm | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| qwen25_vl_7b_local | direct_vlm | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| ollama_llava_13b | direct_vlm | server1 | 0/125 | EXCLUDED | local lane missing |  |
| ollama_qwen3_vl_8b | hybrid | server1 | 0/125 | EXCLUDED | local lane missing |  |
| glm_ocr_qwen3_8b | hybrid | server1 | 123/125 | APPENDIX | local lane incomplete | descriptive_only |
| doctr_qwen3_8b | hybrid | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| easyocr_qwen3_8b | hybrid | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| trocr_qwen3_8b | hybrid | server2 | 125/125 | PRIMARY TABLE |  | wilcoxon;mcnemar;friedman;bootstrap_ci;holm |
| surya_qwen3_8b | hybrid | server1 | 117/125 | APPENDIX | local lane incomplete | descriptive_only |
| docling_qwen3_8b | hybrid | server1 | 124/125 | APPENDIX | local lane incomplete | descriptive_only |
| qwen3_vl_raw_ocr_qwen3_8b | hybrid | server1 | 119/125 | APPENDIX | local lane incomplete | descriptive_only |
| glm_ocr_qwen25_14b | hybrid | server1 | 92/125 | APPENDIX | local lane incomplete | descriptive_only |
| doctr_qwen25_14b | hybrid | server1 | 108/125 | APPENDIX | local lane incomplete | descriptive_only |
| easyocr_qwen25_14b | hybrid | server1 | 112/125 | APPENDIX | local lane incomplete | descriptive_only |
| trocr_qwen25_14b | hybrid | server1 | 124/125 | APPENDIX | local lane incomplete | descriptive_only |
| surya_qwen25_14b | hybrid | server1 | 102/125 | APPENDIX | local lane incomplete | descriptive_only |
| docling_qwen25_14b | hybrid | server1 | 106/125 | APPENDIX | local lane incomplete | descriptive_only |
| qwen3_vl_raw_ocr_qwen25_14b | hybrid | server1 | 95/125 | APPENDIX | local lane incomplete | descriptive_only |
