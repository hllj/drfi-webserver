import json
import os


class Model_State:
    def __init__(self, model_state_path):
        self.model_state_path = model_state_path
        if os.path.exists(model_state_path) is False:
            print("Not exists model state")
            init_model_state = {
                "simi_finished": False,
                "sal_finished": False,
                "fusion_finished": False,
            }
            with open(self.model_state_path, "w") as f:
                json.dump(init_model_state, f)
        self.model_state = self.get_state()

    def get_state(self):
        with open(self.model_state_path, "r") as f:
            return json.load(f)

    def save_state(self):
        with open(self.model_state_path, "w") as f:
            json.dump(self.model_state, f)

    def get_key(self, key):
        return self.model_state[key]

    def set_key(self, key, value):
        self.model_state[key] = value
        self.save_state()
