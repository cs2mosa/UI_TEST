import time
import random
from app.core.types import VideoChunk, VLMResult

SCENARIOS = [
    ("Worker safely traversing walkway with full PPE equipped.", 0.05),
    ("Forklift moving pallet at controlled low speed in designated lane.", 0.12),
    ("Routine material handling in assembly bay; no immediate hazards.", 0.15),
    ("Pedestrian passing close to active forklift travel lane.", 0.42),
    ("Operator working on elevated surface without fall protection harness secured.", 0.78),
    ("Worker on ladder leaning dangerously near unguarded platform edge.", 0.88),
    ("Worker unhooked on scaffolding near overhead crane load.", 0.92),
    ("Tripping hazard: cables and tools left across primary emergency exit walkway.", 0.65),
]

class MockVLM:
    """Mock VLM callable for rapid UI testing and verification."""
    def __init__(self, simulated_delay_s: float = 0.3):
        self.simulated_delay_s = simulated_delay_s

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        if self.simulated_delay_s > 0:
            time.sleep(self.simulated_delay_s)
        
        # Pick scenario deterministically based on chunk_id for reproducible tests
        caption, score = SCENARIOS[chunk.chunk_id % len(SCENARIOS)]
        return VLMResult(caption=caption, score=score)
