from pipeline.train.fusion import Fusion
from pipeline.train.similarities import Similarities
from pipeline.train.saliency import Saliency


class WorkFlow:
    def __init__(self, similarity: Similarities, saliency: Saliency, fusion: Fusion):
        self.similarity = similarity
        self.saliency = saliency
        self.fusion = fusion

    def execute(self):
        model_simi = self.similarity.execute()
        model_sal = self.saliency.execute(model_simi)
        model_fusion = self.fusion.execute(model_sal)
