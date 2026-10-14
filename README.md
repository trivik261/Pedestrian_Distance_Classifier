# Pedestrian Detection & Distance Visualization

Detect people in an image, draw bounding boxes, and compute unique pairwise distances between box centers.

## Portable entry point
```bash
python -m pip install -r requirements.txt
python pedestrian.py sample/image34.jpg --output annotated.jpg
python -m unittest discover -s tests -v
```
The default detector is OpenCV's built-in HOG/SVM person detector, which needs no downloaded weights. For the original SSD MobileNet method, provide the pretrained frozen TensorFlow graph and its matching OpenCV graph configuration:
```bash
python pedestrian.py input.jpg --model frozen_inference_graph.pb --config graph.pbtxt --output annotated.jpg
```
Use `--boxes boxes.json` for detector-independent geometry with supplied `[x,y,width,height]` boxes. `--pixels-per-meter N` enables an explicitly calibrated approximation; otherwise distances are **pixels**. A single image does not establish physical distances or safety thresholds. The historical notebook's `(width-150)/7` assumption is not applied in the portable implementation.

## Fixes
Removed Colab/Drive path dependencies from the run path, avoided division by zero for vertically aligned centers, replaced duplicate permutations with unique combinations, and handled empty detections. The notebook preserves the original TensorFlow 1.x/SSD workflow for historical context. HOG and SSD are different detectors; do not compare their outputs as though they were the same experiment.
