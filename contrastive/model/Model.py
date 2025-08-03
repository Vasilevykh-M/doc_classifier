import torch
from collections import Counter

from model.ORTModel import ORTModel
from model.TRTModel import TRTModel
from model.TorchModel import TorchModel

class Model:
    def __init__(self, model_path, format = "torch", dict_path = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if format == "torch":
            self.model = TorchModel(model_path)
        if format == "ort":
            self.model = ORTModel(model_path)
        if format == "trt":
            self.model = TRTModel(model_path)
        dict_ = [{}, {}] if dict_path is None else torch.load(dict_path)
        self.dict = dict_[0]
        self.idx_to_class = dict_[1]
        for key in self.dict:
            self.dict[key].to(self.device)
    
    def build_dict(self, images, labels, dict_path, idx_to_class):
        val = Counter(labels).keys()
        count = Counter(labels).values()
        for image, label in zip(images, labels):
            embeding = self.model(image)[0]
            if not label is self.dict:
                self.dict[label] = torch.zeros(768).to(self.device)
            self.dict[label] += embeding
        
        for (v, c) in zip(val, count):
            self.dict[v] /= c
        self.idx_to_class = idx_to_class
        torch.save([self.dict, idx_to_class], dict_path)

    def __call__(self, x):
        embeding = self.model(x)[0]
        distances = {country: torch.nn.functional.cosine_similarity(embeding, centroid, dim=0) 
                    for country, centroid in self.dict.items()}
        predicted_country = max(distances, key=distances.get)
        predicted_country = self.idx_to_class[predicted_country]
        return predicted_country