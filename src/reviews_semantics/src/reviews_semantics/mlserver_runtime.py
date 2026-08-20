from __future__ import annotations

from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

from mlserver import MLModel
from reviews_semantics.registry import load_default_predictor


class ReviewScorePredictorRuntime(MLModel):
    async def load(self) -> bool:
        self._predictor = load_default_predictor()
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        texts = [str(value) for value in payload.inputs[0].data]
        scores = self._predictor.predict(texts)
        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(name="score", shape=[len(scores)], datatype="INT64", data=scores)
            ],
        )