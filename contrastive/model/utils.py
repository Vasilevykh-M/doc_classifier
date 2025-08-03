import open_clip

def get_processor(model_path='tulip-B-16-224.ckpt'):
    _, _, preprocess = open_clip.create_model_and_transforms('TULIP-B-16-224', pretrained=model_path)
    return preprocess