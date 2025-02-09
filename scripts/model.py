import torch.nn as nn
import torchvision.models as models

class MRIClassifier(nn.Module):
    """
    A ResNet-based classifier that outputs:
      1) A 'latent_vector' of size embed_dim
      2) A 2-class (knee vs. brain) prediction
    """
    def __init__(self, num_classes=2, embed_dim=128, pretrained=True):
        super(MRIClassifier, self).__init__()
        
        # Load a pretrained ResNet-18 backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the first conv layer if needed (e.g., if your MRI is single-channel)
        # By default, ResNet expects 3-channel (RGB). If your MRI is 1-channel, do this:
        # only thing changed from default here is the channel size
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract the number of output features from ResNet's penultimate layer
        # Typically, resnet.fc.in_features = 512 for ResNet-18
        in_features = self.resnet.fc.in_features
        
        # Remove the original classification head: (fc): Linear(in_features=512, out_features=1000, bias=True)
        # We'll replace it with two heads: an "embedding head" and a "classifier head"
        self.resnet.fc = nn.Identity()
        
        # Embedding head: projects ResNet features into a smaller latent dimension
        self.embedding_layer = nn.Linear(in_features, embed_dim)
        
        # Classification head: maps the embedding to your classes
        self.classifier_head = nn.Linear(embed_dim, num_classes)
        
        # Optionally, add a non-linear activation after the embedding
        # for improved representational power:
        # self.embedding_activation = nn.ReLU()
        
    def forward(self, x):
        # Forward pass through ResNet (minus its original fc layer)
        features = self.resnet(x)
        
        # Convert to a latent vector
        latent_vector = self.embedding_layer(features)
        # latent_vector = self.embedding_activation(latent_vector)  # if you added a ReLU
        
        # Get classification logits
        logits = self.classifier_head(latent_vector)
        
        return latent_vector, logits

# Instantiate the model
model = MRIClassifier(num_classes=2, embed_dim=128, pretrained=True)
print(model)

print("\n\n\n\n")

model_resnet = models.resnet18(pretrained=True)
print(model_resnet)