# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import json
from pathlib import Path

# Force the project root directory onto the python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.gt_adapter import GTAdapter
from src.adapters.legacy_prediction_adapter import LegacyPredictionAdapter
from src.benchmark.runner import BenchmarkRunner
from src.schemas.raw_extraction import CanonicalRawDoc

def main():
    print("=" * 80)
    print("RUNNING PIECE-WISE PRESCRIPTION PIPELINE SMOKE TESTS (p1/p2 VALIDATION)")
    print("=" * 80)
    
    # 1. Verify rapidfuzz or difflib fallback import works
    try:
        from rapidfuzz import fuzz
        print("[OK] Import check: 'rapidfuzz' is successfully installed and imported.")
    except ImportError:
        print("[WARNING] Import check: 'rapidfuzz' not found. Using 'difflib' fallback mechanism.")

    # 2. Paths
    p1_gt_path = PROJECT_ROOT / "raw_ground_truths" / "p1.json"
    p2_gt_path = PROJECT_ROOT / "raw_ground_truths" / "p2.json"
    
    p1_pred_path = PROJECT_ROOT / "outputs" / "raw_extractions" / "Qwen_Final" / "p1.json"
    p2_pred_path = PROJECT_ROOT / "outputs" / "raw_extractions" / "Qwen_Final" / "p2.json"
    
    # Check paths exist
    for label, path in [
        ("p1 GT", p1_gt_path), ("p2 GT", p2_gt_path),
        ("p1 Pred", p1_pred_path), ("p2 Pred", p2_pred_path)
    ]:
        if not path.exists():
            print(f"[FAIL] Path verification: {label} file not found at '{path}'")
            sys.exit(1)
        print(f"[OK] Path verification: {label} exists.")

    # 3. Load GT and predictions through adapters
    print("\n--- LOADING AND PARSING DOCUMENTS ---")
    try:
        p1_gt = GTAdapter.from_file(p1_gt_path)
        p2_gt = GTAdapter.from_file(p2_gt_path)
        print("[OK] GTAdapter successfully loaded both ground truth documents.")
        assert isinstance(p1_gt, CanonicalRawDoc)
        assert isinstance(p2_gt, CanonicalRawDoc)
    except Exception as e:
        print(f"[FAIL] GTAdapter loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    try:
        p1_pred = LegacyPredictionAdapter.from_file(p1_pred_path)
        p2_pred = LegacyPredictionAdapter.from_file(p2_pred_path)
        print("[OK] LegacyPredictionAdapter successfully loaded both Qwen predictions.")
        assert isinstance(p1_pred, CanonicalRawDoc)
        assert isinstance(p2_pred, CanonicalRawDoc)
    except Exception as e:
        print(f"[FAIL] LegacyPredictionAdapter loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Write temporary manifest for p1/p2 smoke run
    manifest_smoke_path = PROJECT_ROOT / "data" / "manifest_smoke.csv"
    manifest_smoke_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_smoke_path, "w", encoding="utf-8") as f:
        f.write("document_id,image_path,gt_path,prediction_path,patient_id,prescription_id,page_number,split\n")
        f.write(f"p1,prescriptions/p1.jpeg,raw_ground_truths/p1.json,outputs/raw_extractions/Qwen_Final/p1.json,pat_01,rx_01,1,validation\n")
        f.write(f"p2,prescriptions/p2.jpeg,raw_ground_truths/p2.json,outputs/raw_extractions/Qwen_Final/p2.json,pat_02,rx_02,1,validation\n")
        
    print(f"[OK] Temporary manifest created at: {manifest_smoke_path}")

    # 5. Run end-to-end evaluation runner on p1 and p2 manifest
    print("\n--- EXECUTING BENCHMARK RUNNER ---")
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom runner run to inspect results in memory
    try:
        runner = BenchmarkRunner(
            manifest_path=manifest_smoke_path,
            project_root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "configs" / "benchmark_defaults.yaml"
        )
        
        # We manually inline parts of the runner run to verify the specific behavioral assertions!
        df = pd_read_manifest(manifest_smoke_path)
        results = []
        
        # Document 1: p1
        p1_res = runner.scalar_matcher.match_docs(p1_gt, p1_pred)
        p1_alignments = []
        categories = [
            ("complaints_or_diagnosis", "complaints_or_diagnosis"),
            ("observations", "observations"),
            ("medications", "medications"),
            ("procedures", "procedures"),
            ("advice", "advice"),
            ("allergy_mentions", "allergy_mentions"),
            ("other_notes", "other_notes"),
            ("lab_observations", "lab_observations")
        ]
        for cat_key, cat_name in categories:
            gt_list = getattr(p1_gt, cat_key, [])
            pred_list = getattr(p1_pred, cat_key, [])
            aligns = runner.entity_matcher.align_entities(gt_list, pred_list, cat_name)
            p1_alignments.extend(aligns)
            
        p1_hallucinations = runner.hallucination_detector.detect_hallucinations(p1_gt, p1_pred, p1_alignments)
        
        # Document 2: p2
        p2_res = runner.scalar_matcher.match_docs(p2_gt, p2_pred)
        p2_alignments = []
        for cat_key, cat_name in categories:
            gt_list = getattr(p2_gt, cat_key, [])
            pred_list = getattr(p2_pred, cat_key, [])
            aligns = runner.entity_matcher.align_entities(gt_list, pred_list, cat_name)
            p2_alignments.extend(aligns)
            
        p2_hallucinations = runner.hallucination_detector.detect_hallucinations(p2_gt, p2_pred, p2_alignments)
        
        print("[OK] Matchers ran successfully without crashing.")
        
        # Execute the full run to dump the files
        summary = runner.run(output_dir=reports_dir)
        print("[OK] BenchmarkRunner saved all report files.")
        
    except Exception as e:
        print(f"[FAIL] Running evaluation orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Specific Behavioral Assertions
    print("\n--- RUNNING SPECIFIC BEHAVIORAL CHECKS ---")
    
    # Requirement A: Bioflu (p1) is a partial medication match
    bioflu_match = [a for a in p1_alignments if a.category == "medications" and "Bioflu" in (a.gt_raw_text or "")]
    if not bioflu_match:
        print("[FAIL] Assertion A: Bioflu medication entry was not aligned or matched.")
        sys.exit(1)
        
    bioflu_align = bioflu_match[0]
    print(f"   Bioflu Match Found: GT='{bioflu_align.gt_raw_text}', Pred='{bioflu_align.pred_raw_text}', Status='{bioflu_align.alignment_status}'")
    assert bioflu_align.alignment_status == "TP_LENIENT", f"Expected TP_LENIENT but got {bioflu_align.alignment_status}"
    print("[PASS] Check A: Bioflu is successfully matched as a partial (lenient) medication.")
    
    # Requirement B: Adv glass is NOT silently corrected (FP in medication, FN in procedure)
    # GT p1 has Adv glasses in procedures, NOT medications.
    # Prediction Qwen p1 predicted it as medication ("items").
    adv_med = [a for a in p1_alignments if a.category == "medications" and "Adv glass" in (a.pred_raw_text or "")]
    adv_proc = [a for a in p1_alignments if a.category == "procedures" and "Adv glass" in (a.gt_raw_text or "")]
    
    if not adv_med or not adv_proc:
        print("[FAIL] Assertion B: Adv glass boundary mapping entries could not be located in medications/procedures.")
        sys.exit(1)
        
    assert adv_med[0].alignment_status == "FP", f"Expected Adv glass prediction to be a False Positive in medications, got {adv_med[0].alignment_status}"
    assert adv_proc[0].alignment_status == "FN", f"Expected Adv glass in GT to be a False Negative in procedures, got {adv_proc[0].alignment_status}"
    
    # Check legacy original details trace
    qwen_items = [m for m in p1_pred.medications if m.raw_name == "Adv glass"]
    assert len(qwen_items) == 1
    assert qwen_items[0].original_category == "items"
    assert qwen_items[0].original_field_path == "items[0]"
    print("[PASS] Check B: 'Adv glass' is correctly NOT silently corrected and reports boundary mismatch (Medication FP, Procedure FN). Original legacy metadata is fully preserved.")
    
    # Requirement C: p2 Kumar Nagar patient name flagged as likely hallucination/OCR confusion
    kumar_record = [h for h in p2_hallucinations if h.category == "patient_info.name" and h.pred_text == "Kumar Nagar"]
    if not kumar_record:
        print("[FAIL] Assertion C: predicted patient name 'Kumar Nagar' was not processed by HallucinationDetector.")
        sys.exit(1)
        
    kumar = kumar_record[0]
    print(f"   Kumar Nagar Flagged: Classification='{kumar.classification}', Rationale='{kumar.rationale}'")
    assert kumar.classification == "likely_hallucination", f"Expected likely_hallucination, got {kumar.classification}"
    assert "Kirti Nagar" in kumar.rationale, f"Expected rationale to trace address 'Kirti Nagar', got '{kumar.rationale}'"
    print("[PASS] Check C: predicted patient name 'Kumar Nagar' flagged as likely hallucination/OCR confusion with address Kirti Nagar.")
    
    # Requirement D: Mogy e/d (RE) qid is evaluated by medication components
    mogy_gt = [m for m in p2_gt.medications if m.raw_name == "Mogy e/d (RE)"]
    if mogy_gt:
        # Check components are populated
        mogy = mogy_gt[0]
        assert mogy.raw_name == "Mogy e/d (RE)" or mogy.raw_name == "Mogy"
        assert mogy.raw_route == "e/d"
        assert mogy.raw_frequency == "qid"
        print("[PASS] Check D: Mogy e/d (RE) qid components parsed successfully into GT medications subfields.")
    else:
        print("[FAIL] Assertion D: Mogy e/d (RE) qid medication not found in ground truth.")
        sys.exit(1)
        
    # Requirement E: Missing observations (IOP and visual acuity) appear as false negatives
    # In p2 GT, observations has: IOP Right Eye 21mmHg, Distant Vision 2/60 etc.
    # In p2 Pred, they are not extracted in observations.
    fn_observations = [a for a in p2_alignments if a.category == "observations" and a.alignment_status == "FN"]
    print(f"   False Negative Observations Found: {len(fn_observations)}")
    assert len(fn_observations) > 0, "Expected missing observations to be flagged as False Negatives."
    print("[PASS] Check E: Missing observations (IOP / visual acuity) successfully registered as False Negatives.")

    # Requirement F: Verify the existence of the 5 requested reports
    print("\n--- VERIFYING FILE GENERATION ---")
    required_reports = [
        "per_document_scores.csv",
        "per_field_scores.csv",
        "entity_alignment_details.csv",
        "manual_review_queue.csv",
        "summary_metrics.json"
    ]
    for filename in required_reports:
        file_path = reports_dir / filename
        if not file_path.exists():
            print(f"[FAIL] Report Verification: Required file '{filename}' was not generated under reports/.")
            sys.exit(1)
        print(f"[OK] Report Verification: '{filename}' generated successfully ({file_path.stat().st_size} bytes).")
    
    print("[PASS] Check F: All required p1/p2 benchmark reports were generated successfully.")

    print("\n" + "=" * 80)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! Exit code 0.")
    print("=" * 80)
    sys.exit(0)

def pd_read_manifest(path: Path):
    # Minimal CSV parser using python standard library to prevent pandas dependency issues
    import csv
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

if __name__ == "__main__":
    main()
