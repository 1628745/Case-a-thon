import torch
import json


model_path = "classification_model.pt"
out_path = "classes.txt"


def load_model():
    checkpoint = torch.load(model_path)

    model = checkpoint['model']
    
    return model


model = load_model()

def classify(some_image):

    out = some_image(model)
    out = out.tolist()
    labels = ["label1", "label2", "label3", "label4", "label5", "label6", "label7"]

    assert len(out) == len(labels)

    mydict = {x:y for x, y in zip(labels, out)}

    with open(out_path, mode='w') as f:
        json.dump(mydict, f)