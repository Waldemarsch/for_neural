import os
import torch

#Конфиг-файл с различными настройками
DATASET_SN6_PATH = os.path.join("./", "datasets/AOI_11_Rotterdam")
DATASET_SN7_PATH = os.path.join("./", "datasets/SN7")
DATASET_BSD_PATH = os.path.join("./", "datasets/BuildingSegmentationDataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
#PIN_MEMORY = True if DEVICE == "cuda" else False
PIN_MEMORY = False
NUM_CHANNELS = 6
NUM_CLASSES = 1
NUM_LEVELS = 6
INIT_LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
TEST_SPLIT = 0.25
THRESHOLD = 0.5
MODEL_OUTPUT = "models"
MODEL_PATH = os.path.join(MODEL_OUTPUT, "unet_forest_segmentation.pth")