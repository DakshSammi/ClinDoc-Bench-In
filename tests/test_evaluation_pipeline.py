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

import pytest
from src.schemas.raw_extraction import CanonicalRawDoc, PatientInformation, RawEntityItem, RawMedicationItem
from src.benchmark.entity_match import EntityMatcher
from src.benchmark.hallucination import HallucinationDetector
from src.benchmark.scalar_match import ScalarMatcher

def test_p1_medication_partial_match():
    """
    Test that Bioflu e/d matches as a partial medication match (TP_LENIENT)
    when only 'Bioflu' is extracted.
    """
    matcher = EntityMatcher()
    
    gt_med = RawMedicationItem(
        raw_line_text="Bioflu e/d",
        raw_name="Bioflu",
        raw_route="e/d",
        raw_frequency="tds"
    )
    
    pred_med = RawMedicationItem(
        raw_line_text="Bioflu",
        raw_name="Bioflu"
    )
    
    alignments = matcher.align_entities([gt_med], [pred_med], category="medications")
    
    assert len(alignments) == 1
    alignment = alignments[0]
    assert alignment.alignment_status == "TP_LENIENT"
    assert alignment.exact_match is False
    assert alignment.lenient_match is True
    # Verify name component scores 1.0 (exact match on "Bioflu") while other components score 0.0 or 1.0 (if both empty)
    assert alignment.medication_subfields is not None
    assert alignment.medication_subfields["name"] == 1.0
    assert alignment.medication_subfields["route"] == 0.0

def test_p1_procedure_boundary():
    """
    Test that 'Adv glasses' is evaluated under procedures, not medications.
    If the model extracts it as a medication, it must register as a
    medication false positive (FP) and a procedure false negative (FN).
    """
    matcher = EntityMatcher()
    
    # Ground Truth: 'Adv glasses' is a procedure
    gt_doc = CanonicalRawDoc(
        document_id="p1",
        procedures=[RawEntityItem(raw_text="Adv glasses", evidence_text="Adv glasses")]
    )
    
    # Prediction: Model extracts it under medications (legacy items default category)
    pred_doc = CanonicalRawDoc(
        document_id="p1",
        medications=[RawMedicationItem(raw_line_text="Adv glasses", raw_name="Adv glasses")]
    )
    
    # Match medications
    med_alignments = matcher.align_entities(gt_doc.medications, pred_doc.medications, category="medications")
    assert len(med_alignments) == 1
    assert med_alignments[0].alignment_status == "FP"
    assert med_alignments[0].pred_raw_text == "Adv glasses"
    
    # Match procedures
    proc_alignments = matcher.align_entities(gt_doc.procedures, pred_doc.procedures, category="procedures")
    assert len(proc_alignments) == 1
    assert proc_alignments[0].alignment_status == "FN"
    assert proc_alignments[0].gt_raw_text == "Adv glasses"

def test_p2_patient_name_hallucination():
    """
    Test that predicting 'Kumar Nagar' as the patient name in p2 is flagged
    as a likely hallucination/OCR confusion because the GT patient name is
    empty and the GT address is 'Kirti Nagar, Sirsa'.
    """
    detector = HallucinationDetector()
    
    gt_doc = CanonicalRawDoc(
        document_id="p2",
        patient_information=PatientInformation(
            name="",
            address="Kirti Nagar, Sirsa"
        )
    )
    
    pred_doc = CanonicalRawDoc(
        document_id="p2",
        patient_information=PatientInformation(
            name="Kumar Nagar"
        )
    )
    
    records = detector.detect_hallucinations(gt_doc, pred_doc, entity_alignments=[])
    
    # Check that patient name hallucination was detected
    name_records = [r for r in records if r.category == "patient_info.name"]
    assert len(name_records) == 1
    record = name_records[0]
    assert record.classification == "likely_hallucination"
    assert "Kirti Nagar" in record.rationale
    assert record.matched_snippet == "Kirti Nagar, Sirsa"

def test_mogy_component_matching():
    """
    Test that 'Mogy e/d (RE) qid' is evaluated properly by checking similarity of its
    components: name (Mogy), route (e/d), laterality/instructions, and frequency (qid).
    """
    matcher = EntityMatcher()
    
    gt_med = RawMedicationItem(
        raw_line_text="Mogy e/d (RE) qid",
        raw_name="Mogy",
        raw_route="e/d",
        raw_frequency="qid",
        raw_instruction="(RE)"
    )
    
    pred_med = RawMedicationItem(
        raw_line_text="Mogy e/d (RE) qid",
        raw_name="Mogy",
        raw_route="e/d",
        raw_frequency="qid",
        raw_instruction="(RE)"
    )
    
    # Perfect match
    sim_perfect, subfields_perfect = matcher.compute_medication_similarity(gt_med, pred_med)
    assert sim_perfect == 1.0
    assert subfields_perfect["name"] == 1.0
    assert subfields_perfect["route"] == 1.0
    assert subfields_perfect["frequency"] == 1.0
    
    # Mismatched components
    pred_med_mismatch = RawMedicationItem(
        raw_line_text="Mogy drops (LE) tds",
        raw_name="Mogy",
        raw_route="drops",
        raw_frequency="tds",
        raw_instruction="(LE)"
    )
    
    sim_mismatch, subfields_mismatch = matcher.compute_medication_similarity(gt_med, pred_med_mismatch)
    assert sim_mismatch < 1.0
    assert subfields_mismatch["name"] == 1.0
    assert subfields_mismatch["route"] < 1.0
    assert subfields_mismatch["frequency"] < 0.5
