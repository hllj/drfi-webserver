import os
import json

from .model_state import Model_State


class Validate_State(Model_State):
    def __init__(self, model_state_path):
        self.model_state_path = model_state_path
        if os.path.exists(model_state_path) is False:
            print("Not exists model state")
            init_validate_state = {
                "simi_finished": False,
                "simi_idx": 0,
                "sal_finished": False,
                "sal_idx": 0,
                "fusion_finished": False,
                "fusion_idx": 0,
            }
            with open(self.model_state_path, "w") as f:
                json.dump(init_validate_state, f)
        self.model_state = self.get_state()
