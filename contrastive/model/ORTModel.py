from pathlib import Path
import open_clip
import torch
from model.utils import get_processor
import onnxruntime as ort
from PIL import Image

class ORTModel:
    def __init__(self, model_path):
        model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_path.suffix != ".onnx" or not model_path.exists():
           model_path = self.build(model_path) 
        self.session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        self.processor = get_processor()

    def preprocess(self, image_path):
        x = Image.open(image_path)
        x = self.processor(x)[None, :].numpy()
        return x

    def __call__(self, image_path):
        inputs = self.preprocess(image_path)
        outputs = self.session.run(None, {"input": inputs})[0]
        return torch.from_numpy(outputs).to(self.device)
    
    def build(self, model_path):
        if model_path.suffix != ".pt" or not model_path.exists():
           model, _ = self.build_model_ckpt(model_path)
        else:
            model = self.build_model_pt(model_path)

        model, _, _ = open_clip.create_model_and_transforms('TULIP-B-16-224', pretrained=model_path)
        dummy_input = torch.rand((1, 3, 224, 224)).to(self.device)
        torch.onnx.export(
            model.to(self.device),
            dummy_input,
            f"{model_path.name}.onnx",
            input_names=["input"],
            output_names=["output"],
        )

        return f"{model_path.name}.onnx"
    
    def build_model_ckpt(self, model_path='contrastive/tulip-B-16-224.ckpt'):
        model, _, preprocess = open_clip.create_model_and_transforms('TULIP-B-16-224', pretrained=model_path)
        return model.visual, preprocess
    
    def build_model_pt(self, model_path):
        model = open_clip.transformer.VisionTransformer()
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model