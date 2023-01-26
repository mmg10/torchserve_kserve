import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet34(weights=None)
model.fc = torch.nn.Linear(512,6)
model.load_state_dict(torch.load("model.bin"), map_location=torch.device('cpu'))
    
# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_model.pt")