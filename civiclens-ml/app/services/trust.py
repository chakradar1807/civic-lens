"""Dynamic Trust Score (0-1) for reporters.

score = 0.5 * identity_prior + 0.5 * smoothed_acceptance_rate - penalty_for_fake_reports
  identity_prior : 0.8 for government-ID-verified users, 0.4 for guests
  acceptance rate: Laplace-smoothed accepted/total, so new users start near 0.5
"""
from typing import Dict


def compute_trust(verified_id: bool, total_reports: int, accepted_reports: int, rejected_fake_reports: int) -> Dict:
    total = max(0, int(total_reports))
    accepted = max(0, min(int(accepted_reports), total))
    fake = max(0, int(rejected_fake_reports))

    prior = 0.8 if verified_id else 0.4
    acceptance = (accepted + 1.0) / (total + 2.0)
    score = 0.5 * prior + 0.5 * acceptance - 0.08 * fake
    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.75:
        tier, action = "HIGH", "Auto-dispatch allowed when AI confidence is sufficient"
    elif score >= 0.5:
        tier, action = "MEDIUM", "Dispatch only if AI confidence is high or a second report confirms"
    else:
        tier, action = "LOW", "Throttle submissions; require a second independent report before dispatch"

    return {
        "trust_score": score,
        "trust_percent": int(round(score * 100)),
        "tier": tier,
        "recommended_action": action,
        "throttle": score < 0.3,
        "components": {"identity_prior": prior, "smoothed_acceptance_rate": round(acceptance, 3), "fake_report_penalty": round(0.08 * fake, 3)},
    }
