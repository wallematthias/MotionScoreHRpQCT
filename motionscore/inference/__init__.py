from motionscore.inference.scoring import PredictionResult, predict_scan

__all__ = ["ModelEnsemble", "PredictionResult", "predict_scan"]


def __getattr__(name: str):
    if name == "ModelEnsemble":
        from motionscore.inference.model import ModelEnsemble

        return ModelEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
