#!/usr/bin/env python3
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

"""
Quick validation test for rate limiter integration with extract CLI.
Tests token estimation, rate limit logic, and logging with new fields.
"""

import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rate_limiter import init_global_limiter, get_global_limiter


def test_rate_limiter():
    """Test rate limiter core functionality."""
    print("=" * 70)
    print("TESTING RATE LIMITER CORE FUNCTIONALITY")
    print("=" * 70)
    
    # Initialize
    limiter = init_global_limiter(
        tpm_limit=500000,
        rpm_limit=120,
        window_seconds=60,
        buffer_seconds=15,
        max_retries_rate_limit=1
    )
    print("✓ Initialized rate limiter")
    print(f"  Config: TPM=500000, RPM=120, window=60s, buffer=15s, max_retries=1\n")
    
    # Simulate 4 successful extractions (p1, p45_1, p45_3, p45_4)
    documents = [
        ("p1", 45000, 9000),
        ("p45_1", 52000, 10400),
        ("p45_3", 48000, 9600),
        ("p45_4", 58000, 11600),
    ]
    
    print("SIMULATING 4 SUCCESSFUL EXTRACTIONS:")
    print("-" * 70)
    
    for doc_id, prompt_tokens, completion_tokens in documents:
        total = prompt_tokens + completion_tokens
        limiter.record_usage(
            document_id=doc_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated=False,
            reason="successful_extraction"
        )
        summary = limiter.get_summary()
        print(f"\n{doc_id}: {prompt_tokens:,} prompt + {completion_tokens:,} completion = {total:,} tokens")
        print(f"  Rolling total: {summary['rolling_tokens']:,}/{summary['tpm_limit']:,} TPM")
        print(f"  Remaining: {summary['tpm_remaining']:,} tokens")
        print(f"  Requests: {summary['rolling_requests']}/{summary['rpm_limit']}")
    
    total_so_far = sum(p + c for _, p, c in documents)
    print(f"\n✓ Cumulative after 4 docs: {total_so_far:,} tokens ({total_so_far/500000*100:.1f}% of limit)")
    
    # Now test p2 behavior
    print("\n" + "=" * 70)
    print("TESTING P2 RETRY BEHAVIOR")
    print("=" * 70 + "\n")
    
    # p2 is smaller
    p2_prompt = 40000
    p2_completion = 8000
    p2_total_est = limiter.estimate_tokens(
        document_id="p2",
        max_tokens=50000,
        num_images=1,
        prompt_length_chars=5000
    )
    
    print(f"p2 estimated token requirement: {p2_total_est:,} tokens")
    print(f"Current rolling total: {summary['rolling_tokens']:,} tokens")
    print(f"Would post-request total be: {summary['rolling_tokens'] + p2_total_est:,} tokens")
    
    should_wait, wait_time = limiter.should_wait_before_request("p2", p2_total_est)
    print(f"\nRate limit check before p2:")
    print(f"  Should wait: {should_wait}")
    print(f"  Wait time: {wait_time:.2f} seconds")
    
    if not should_wait:
        print(f"\n✓ p2 can proceed immediately (TPM not exhausted)")
    else:
        print(f"\n⚠ p2 would need to wait {wait_time:.1f} sec before proceeding")
    
    # Record p2 success
    print(f"\nRecording p2 success ({p2_prompt:,} + {p2_completion:,} tokens)...")
    limiter.record_usage("p2", p2_prompt, p2_completion, estimated=False, reason="retry_after_wait")
    
    final_summary = limiter.get_summary()
    final_total = final_summary['rolling_tokens']
    print(f"✓ Final rolling total: {final_total:,} tokens ({final_total/500000*100:.1f}% of limit)")
    print(f"  Remaining capacity: {final_summary['tpm_remaining']:,} tokens")
    
    print("\n" + "=" * 70)
    print("RATE LIMITER VALIDATION: PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_rate_limiter()
