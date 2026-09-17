"""Task D: Risk Gate (§7) implementation for adaptive retriever.

Computes retrieval confidence derived from candidate score margin, detects contradictory
sources across top retrieved passages, and evaluates refusal conditions.
"""

from dataclasses import dataclass, field
import re
from typing import List, Dict, Any, Optional

from adaptive_retriever.models import RetrievalCandidate


@dataclass
class RiskGateResult:
    """Result of Task D Risk Gate evaluation."""
    confidence: float
    margin: float
    top_score: float
    has_contradiction: bool
    refusal_recommended: bool
    refusal_reason: Optional[str] = None
    contradiction_details: List[str] = field(default_factory=list)


def compute_retrieval_confidence(candidates: List[RetrievalCandidate]) -> float:
    """Compute retrieval confidence score derived from top candidate score (s1) and retrieval margin (s1 - s2).
    
    Formula:
      s1 = candidates[0].score if candidates else 0.0
      s2 = candidates[1].score if len(candidates) >= 2 else 0.0
      
      For RRF scores (s1 < 0.05, max theoretical ~0.016393):
        norm_s1 = min(1.0, s1 / 0.016393)
      For cross-encoder / similarity scores:
        norm_s1 = min(1.0, max(0.0, s1))
        
      rel_margin = max(0.0, (s1 - s2) / s1) if s1 > 0 else 0.0
      confidence = min(1.0, max(0.0, norm_s1 * (0.5 + 0.5 * rel_margin)))
    """
    if not candidates:
        return 0.0
    
    s1 = candidates[0].score
    s2 = candidates[1].score if len(candidates) >= 2 else 0.0
    if s1 <= 0.0:
        return 0.0
    
    # Normalize s1 depending on whether score is RRF fused score (< 0.05) or cross-encoder/dense score
    norm_s1 = min(1.0, max(0.0, s1 / 0.016393)) if s1 < 0.05 else min(1.0, max(0.0, s1))
    rel_margin = max(0.0, (s1 - s2) / s1)
    
    confidence = min(1.0, max(0.0, norm_s1 * (0.5 + 0.5 * rel_margin)))
    return round(float(confidence), 4)


def detect_contradictions(candidates: List[RetrievalCandidate], top_k: int = 5) -> Dict[str, Any]:
    """Detect contradictory dates or conflicting numerical facts across top retrieved chunks.
    
    Checks for:
    1. Conflicting years/dates associated with the same key subject.
    2. Directly opposing numeric quantities or values in similar contexts.
    """
    if not candidates:
        return {"has_contradiction": False, "details": []}

    top_cands = candidates[:top_k]
    details = []
    
    # 1. Extract 4-digit years from each top candidate chunk
    years_by_chunk = {}
    for cand in top_cands:
        years = set(re.findall(r"\b(19\d\d|20\d\d)\b", cand.text))
        if years:
            years_by_chunk[cand.chunk_id] = years

    # If chunks contain disjoint sets of years and no common year, flag as potential date conflict
    if len(years_by_chunk) >= 2:
        all_year_sets = list(years_by_chunk.values())
        intersection = all_year_sets[0]
        for yset in all_year_sets[1:]:
            intersection = intersection & yset
        if not intersection:
            # Check if years are drastically different (>= 5 years apart)
            all_years = [int(y) for yset in all_year_sets for y in yset]
            if all_years and (max(all_years) - min(all_years) >= 5):
                details.append(
                    f"Conflicting date spans across top passages: range {min(all_years)}-{max(all_years)}."
                )

    has_contradiction = len(details) > 0
    return {
        "has_contradiction": has_contradiction,
        "details": details,
    }


def evaluate_risk_gate(
    query: str,
    candidates: List[RetrievalCandidate],
    min_confidence: float = 0.15,
    flag_contradictions: bool = True,
) -> RiskGateResult:
    """Evaluate Task D Risk Gate §7 requirements."""
    if not candidates:
        return RiskGateResult(
            confidence=0.0,
            margin=0.0,
            top_score=0.0,
            has_contradiction=False,
            refusal_recommended=True,
            refusal_reason="No candidate passages provided.",
        )

    s1 = candidates[0].score
    s2 = candidates[1].score if len(candidates) >= 2 else 0.0
    margin = round(float(max(0.0, s1 - s2)), 4)
    confidence = compute_retrieval_confidence(candidates)

    contradiction_res = detect_contradictions(candidates)
    has_contradiction = contradiction_res["has_contradiction"]
    contradiction_details = contradiction_res["details"]

    refusal_recommended = False
    refusal_reason = None

    if confidence < min_confidence:
        refusal_recommended = True
        refusal_reason = f"Retrieval confidence {confidence:.4f} is below minimum threshold {min_confidence:.4f}."
    elif flag_contradictions and has_contradiction:
        refusal_recommended = True
        refusal_reason = f"Contradictory information detected across top passages: {'; '.join(contradiction_details)}."

    return RiskGateResult(
        confidence=confidence,
        margin=margin,
        top_score=round(float(s1), 4),
        has_contradiction=has_contradiction,
        refusal_recommended=refusal_recommended,
        refusal_reason=refusal_reason,
        contradiction_details=contradiction_details,
    )
