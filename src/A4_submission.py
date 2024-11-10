import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import yaml

# Load data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Train
import torch.optim as optim
from yolov5.models.yolo import Model
from yolov5.utils.loss import ComputeLoss
from tqdm import tqdm
from yolov5.utils.metrics import box_iou
from yolov5.utils.general import non_max_suppression

# View results
from yolov5.utils.general import non_max_suppression

class Args:
    gen_data = False
    train = True
    mode = "train"
    model = "ultralytics/yolov5"

    # Training parameters
    batch_size = 16
    hyp = {
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.2,  # Final learning rate (as a fraction of lr0)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # L2 regularization (weight decay)
        'warmup_epochs': 3.0,  # Warmup duration in epochs
        'warmup_bias_lr': 0.1,  # Warmup initial bias learning rate
        'box': 0.05,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'obj': 1.0,  # Objectness loss gain (used for object confidence)
        'label_smoothing': 0.0,  # Label smoothing (set between 0-1)
        'iou_t': 0.20,  # IoU threshold for positive predictions
        'anchor_t': 4.0,  # Anchor-matching threshold
        'fl_gamma': 0.0,  # Focal loss gamma (if > 0, use focal loss for class balance)
        'hsv_h': 0.015,  # Image HSV-Hue augmentation
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation
        'hsv_v': 0.4,  # Image HSV-Value augmentation
        'degrees': 0.0,  # Image rotation augmentation (degrees)
        'translate': 0.1,  # Image translation augmentation (fraction)
        'scale': 0.5,  # Image scale augmentation (fraction)
        'shear': 0.0,  # Image shear augmentation
        'perspective': 0.0,  # Perspective augmentation
        'flipud': 0.0,  # Vertical flip probability
        'fliplr': 0.5,  # Horizontal flip probability
        'mosaic': 1.0,  # Mosaic augmentation probability
        'mixup': 0.0,  # MixUp augmentation probability
        'copy_paste': 0.0,  # Copy-paste augmentation probability
        'cls_pw': 1.0,          # Class positive weight
        'obj_pw': 1.0,          # Object positive weight
        'anchors': 3.0          # Anchor multiple (used in anchor generation)
    }

class MNISTDDDataset(Dataset):
    def __init__(self, images, labels, bboxes):
        """
        :param images: Flattened images of shape (N, 12288)
        :param labels: Class labels of shape (N, 2)
        :param bboxes: Bounding boxes of shape (N, 2, 4)
        """
        self.images = images.reshape(-1, 3, 64, 64).astype('float32') / 255.0  # Normalize images to (C, H, W)
        self.labels = labels
        self.bboxes = bboxes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])                  # Convert to tensor, shape (C, H, W)
        label = self.labels[idx]
        bbox = self.bboxes[idx]

        # YOLOv5 expects targets in [image_index, class_id, x_center, y_center, width, height]
        targets = torch.zeros((len(bbox), 6))  # [batch_index, class, x_center, y_center, width, height]
        targets[:, 1] = torch.tensor(label, dtype=torch.float32)    # Class IDs
        targets[:, 2:] = self.convert_bboxes_to_yolo(bbox)          # YOLO bounding boxes

        return image, targets

    def convert_bboxes_to_yolo(self, bboxes):
        """
        Convert bounding boxes from [xmin, ymin, xmax, ymax] to YOLO format [x_center, y_center, width, height].
        """
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2 / 64
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2 / 64
        width = (bboxes[:, 2] - bboxes[:, 0]) / 64
        height = (bboxes[:, 3] - bboxes[:, 1]) / 64
        return torch.stack([x_center, y_center, width, height], dim=1)

# Load and modify YOLOv5 configuration
def load_custom_yolo_model():
    model_config = "yolov5s.yaml"
    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    cfg['nc'] = 10  # Set number of classes to 10 for MNISTDD
    model = Model(cfg)
    return model

def load_dataset(mode):
    '''
    Return a Dataset object for a given mode
    :param mode: 'train' or 'test'
    :return: Dataset object for a given mode
    '''
    mnistdd_data = np.load(f"{mode}.npz")
    images = mnistdd_data['images']
    labels = mnistdd_data['labels']
    bboxes = mnistdd_data['bboxes']
    dataset = MNISTDDDataset(images, labels, bboxes)
    return dataset

def collate_fn(batch):
    '''
    Used by YOLOv5 data loader for batching
    '''
    images, targets = zip(*batch)
    for i, target in enumerate(targets):
        target[:, 0] = i  # Set batch index
    return torch.stack(images), torch.cat(targets, dim=0)

def train_mnistdd(hyp, train_loader, val_loader, model, device, epochs=3, save_dir='./ckpt'):
    """
    Train YOLOv5 model on MNISTDD dataset.
    :param hyp: Hyperparameter dictionary
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param model: YOLOv5 model
    :param device: Training device (CPU/GPU)
    :param epochs: Number of training epochs
    :param save_dir: Directory for saving checkpoints
    """
    # Attach hyperparameters to the model
    model.hyp = hyp

    compute_loss = ComputeLoss(model)  # YOLOv5's loss function
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrf'], steps_per_epoch=len(train_loader), epochs=epochs)

    best_map = 0.0
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')

        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, _ = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (i + 1)})

        # Validation after each epoch
        val_map = validate(val_loader, model, device)
        print(f'Epoch {epoch+1}/{epochs} | Training Loss: {epoch_loss/len(train_loader):.4f} | Validation mAP: {val_map:.4f}')

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), f'{save_dir}/best.pt')
            print(f"New best model saved with mAP: {best_map:.4f}")

    # Save the final model
    torch.save(model.state_dict(), f'{save_dir}/last.pt')

def validate(val_loader, model, device, iou_threshold=0.5, conf_threshold=0.25):
    """
    Validate the model on the validation set.
    :param val_loader: DataLoader for validation data
    :param model: YOLOv5 model
    :param device: Device to use ('cuda' or 'cpu')
    :param iou_threshold: IoU threshold for NMS
    :param conf_threshold: Confidence threshold for valid detections
    :return: Mean Average Precision (mAP)
    """
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            preds = model(images)  # Get predictions

            # Apply NMS to filter predictions
            preds = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=iou_threshold)

            # Convert predictions and targets to standard formats
            for pred, target in zip(preds, targets):
                if pred is not None:
                    pred_boxes = pred[:, :4].cpu().numpy()  # Predicted boxes (x1, y1, x2, y2)
                    pred_scores = pred[:, 4].cpu().numpy()  # Confidence scores
                    pred_labels = pred[:, 5].cpu().numpy()  # Predicted classes
                    all_pred_boxes.append((pred_boxes, pred_scores, pred_labels))
                else:
                    all_pred_boxes.append((np.array([]), np.array([]), np.array([])))

                # Convert targets to the format (x1, y1, x2, y2, class)
                target_boxes = target[:, 2:].cpu().numpy()  # Ground truth boxes
                target_labels = target[:, 1].cpu().numpy()  # Ground truth classes
                all_true_boxes.append((target_boxes, target_labels))

    # Calculate mean Average Precision (mAP)
    return calculate_map(all_pred_boxes, all_true_boxes, iou_threshold)

def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=10):
    """
    Calculate mean Average Precision (mAP) for predicted and true boxes.
    :param pred_boxes: List of tuples (pred_boxes, scores, pred_labels) for each image
    :param true_boxes: List of tuples (true_boxes, true_labels) for each image
    :param iou_threshold: IoU threshold for matching boxes
    :param num_classes: Number of classes
    :return: Mean Average Precision (mAP)
    """
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for (pred_boxes, pred_scores, pred_labels), (true_boxes, true_labels) in zip(pred_boxes, true_boxes):
            # Filter by class
            pred_mask = pred_labels == c
            true_mask = true_labels == c

            detections.extend(list(zip(pred_scores[pred_mask], pred_boxes[pred_mask])))
            ground_truths.extend(true_boxes[true_mask])

        # Sort detections by confidence
        detections.sort(key=lambda x: x[0], reverse=True)
        detections = [(d[1], d[0]) for d in detections]

        # Calculate AP for this class
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        total_ground_truths = len(ground_truths)

        assigned_gt = []

        for idx, (pred_box, _) in enumerate(detections):
            ious = [box_iou(torch.tensor(pred_box).unsqueeze(0), torch.tensor(gt_box).unsqueeze(0)).item()
                    for gt_box in ground_truths]

            if len(ious) == 0:
                fp[idx] = 1
                continue

            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou > iou_threshold and best_iou_idx not in assigned_gt:
                tp[idx] = 1
                assigned_gt.append(best_iou_idx)
            else:
                fp[idx] = 1

        # Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (total_ground_truths + 1e-6)

        # Interpolated AP calculation
        recalls = np.concatenate(([0], recalls, [1]))
        precisions = np.concatenate(([1], precisions, [0]))
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        average_precisions.append(ap)

    return np.mean(average_precisions)

def detect_and_segment(images):
    """
    :param images: Flattened images of shape (N, 12288)
    :return: Predicted classes, bounding boxes, and segmentation masks
    """
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None

    if args.train:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=10)
        train_dataset = load_dataset("train")
        valid_dataset = load_dataset("valid")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
        train_mnistdd(args.hyp, train_loader, val_loader, model, device)
    else:
        # Load trained YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, classes=10).to(device)
        model.load_state_dict(torch.load('./ckpt/best.pt'))

    model.eval()

    N = images.shape[0]
    reshaped_images = images.reshape(N, 3, 64, 64).astype('float32') / 255.0
    tensor_images = torch.tensor(reshaped_images).to(device)

    pred_class = np.zeros((N, 2), dtype=np.int32)
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    pred_seg = np.zeros((N, 4096), dtype=np.int32)  # Placeholder for segmentation

    with torch.no_grad():
        preds = model(tensor_images)  # YOLOv5 prediction
        detections = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

        for i, det in enumerate(detections):
            if det is not None:
                det = det.cpu().numpy()
                boxes = det[:2, :4]  # Get top 2 bounding boxes
                classes = det[:2, 5].astype(int)  # Get top 2 class labels

                pred_bboxes[i] = boxes
                pred_class[i] = sorted(classes)  # Sort class labels as required

                # Optional: Generate a basic segmentation mask from predicted bounding boxes
                pred_seg[i] = generate_segmentation_mask(reshaped_images[i], boxes)
            else:
                print("Model made no detections")
                pred_bboxes[i] = np.zeros((2, 4))
                pred_class[i] = [0, 1]  # Dummy values for empty predictions

    return pred_class, pred_bboxes, pred_seg

def generate_segmentation_mask(image, boxes):
    """
    Generate a segmentation mask based on predicted bounding boxes.
    :param image: Original image
    :param boxes: Predicted bounding boxes
    :return: Flattened segmentation mask
    """
    mask = np.zeros((64, 64), dtype=np.int32)
    for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes.astype(int)):
        mask[ymin:ymax, xmin:xmax] = idx + 1  # Assign labels 1 and 2 for digits
    return mask.flatten()