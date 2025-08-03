import cv2
from onnxtr.models import ocr_predictor
import re
import json

class Model:
    def __init__(self, country_keywords):

        with open(country_keywords, 'r', encoding='utf-8') as f:
            self.country_keywords = json.load(f)

        self.model = ocr_predictor()
        self.regex = r"[ \n]"

    def preprocess(self, image_path, MRZ):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if MRZ:
            height, _, _ = img.shape
            img = img[height//2:, :]

        return [img]
    
    def predict(self, inputs):
        text = self.model(inputs)
        return text
    
    def postprocess(self, outputs, MRZ):
        outputs = outputs.render()
        if MRZ:
            country_match = re.search(r'P\<[A-Z]{3}', outputs)
            return country_match.group(0)[2:] if country_match else None
        
        outputs = re.split(self.regex, outputs.lower())
        for code, keywords in self.country_keywords.items():
            if any(keyword in outputs for keyword in keywords):
                return code
        return None
    
    def recognition(self, image_path, MRZ):
        inputs = self.preprocess(image_path, MRZ)
        outputs = self.predict(inputs)
        result = self.postprocess(outputs, MRZ)
        return result
    
    def __call__(self, image_path):
        result = self.recognition(image_path, True)
        if not result:
            result = self.recognition(image_path, False)
        return result