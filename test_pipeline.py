import cv2
from sam2aug import AugmentationPipeline, Segmenter, LamaInpainter, Relocator
from sam2aug.config import SAM2_CONFIG, SAM2_CHECKPOINT, LAMA_CONFIG_PATH, LAMA_CHECKPOINT_PATH

image_path = "/data/local/rgaisina/datasets/imagenet/val_classed/n01514859/ILSVRC2012_val_00001368.JPEG"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

boxes = [[1,56,361,498]]

segmenter = Segmenter(
    model_config=SAM2_CONFIG,
    checkpoint_path=SAM2_CHECKPOINT,
    device="cuda",
)

inpainter = LamaInpainter(
    config_path=LAMA_CONFIG_PATH,
    checkpoint_path=LAMA_CHECKPOINT_PATH,
)

relocator = Relocator()

pipeline = AugmentationPipeline(
    segmenter=segmenter,
    inpainter=inpainter,
    relocator=relocator,
    save_intermediate=True,
    save_dir="smoke_test_outputs",
    enable_inpainting=True,
    enable_relocation=True,
)

results = pipeline(
    image_rgb=image_rgb,
    boxes=boxes,
    image_id="smoke_test",
)

print(f"Number of results: {len(results)}")
print(results[0].keys())