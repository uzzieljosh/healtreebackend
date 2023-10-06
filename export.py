from ultralytics import YOLO

# Load a model
model = YOLO('Tilliabest.pt')  # load a custom trained

# Export the model
model.export(format='torchscript')