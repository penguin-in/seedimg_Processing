import sys
sys.path.append(r"/home/liushuai/seed/neural_network/model")
from Resnet import resnet18,resnet34,resnet50,resnet101

model_dict = {
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
    'resnet101':resnet101,
}

def create_model(model_name,num_classes):
    return model_dict[model_name](num_classes = num_classes)