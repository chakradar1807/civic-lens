# Colab / Kaggle training recipe (copy each block into a notebook cell)

```python
!pip install -q ultralytics roboflow
!git clone <your-ml-repo-url> civiclens-ml && cd civiclens-ml
```

```python
# Download datasets in "YOLOv8" format from Roboflow Universe (free API key) - one per class:
# search: "pothole detection", "garbage detection", "street light" / "damaged street light", "flood" / "waterlogging"
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
# rf.workspace("...").project("...").version(1).download("yolov8", location="datasets/potholes")
```

```python
# Merge into 4 classes. Open each source's data.yaml to see its class ids, then map them:
!python training/merge_datasets.py \
   --source datasets/potholes --map pothole=0 \
   --source datasets/garbage  --map garbage=0 \
   --source datasets/lights   --map damaged_streetlight=0 \
   --source datasets/floods   --map waterlogging=0 \
   --out datasets/civiclens
```

```python
!python training/train_yolo.py --data training/data.yaml --epochs 50 --imgsz 640
# Download models/best.pt and runs/civiclens/yolov8n/{confusion_matrix.png,results.png,PR_curve.png,summary.json}
```
