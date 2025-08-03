from pathlib import Path
import open_clip
import torch
from PIL import Image
from loss.ContrastiveLoss import ContrastiveLoss
from model.utils import get_processor

class TorchModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_math = Path(model_path)
        if model_math.suffix != ".pt" or not model_math.exists():
           model, processor = self.build_model_ckpt(model_path)
        else:
            _, processor = get_processor()
            model = self.build_model_pt(model_path)
        self.model = model
        self.model.to(self.device)
        self.processor = processor
        self.loss_ = ContrastiveLoss()

    def build_model_ckpt(self, model_path='contrastive/tulip-B-16-224.ckpt'):
        model, _, preprocess = open_clip.create_model_and_transforms('TULIP-B-16-224', pretrained=model_path)
        return model.visual, preprocess
    
    def build_model_pt(self, model_path):
        model = open_clip.transformer.VisionTransformer()
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    def loss(self, images, targets):
        pred = self.model(images)
        loss = self.loss_(pred, targets)
        return loss
    
    def preprocess(self, image_path):
        x = Image.open(image_path)
        x = self.processor(x).to(self.device)
        return x
    
    def __call__(self, x):
        x = self.preprocess(x)
        embeding = self.model(x[None, :])
        return embeding