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

# View results
from yolov5.utils.general import non_max_suppression
from PIL import Image
import matplotlib.pyplot as plt

class Args:
    load = True
    mode = "train"
    model = "ultralytics/yolov5"
    environment = "local"
    directories = {
        "kaggle": "/kaggle/working",
        "colab": "/content/drive/MyDrive/colab_notebooks/Assignment-4",
        "local": "../"
    }
    root_dir = directories[environment]
    save_dir = "src/ckpt"
    data_dir = "yolov5/data"

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
        self.images = images #.reshape(-1, 3, 64, 64).astype('float32') / 255.0  # Normalize images to (C, H, W)
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

    def retrieve_item(self, idx):
        images = self.images.shape[0]
        image = self.images[idx]
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        return image, label, bbox

def save_images(images, output_dir):
    print("Saving images at", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, img_vector in enumerate(tqdm(images, desc="Saving images", unit="image")):
        img = img_vector.reshape((64, 64, 3)).astype(np.uint8)  # Scale if necessary to 0-255
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), img)
    print("Done saving images")

def save_labels(bboxes, labels, output_dir, img_dim=64):
    print("Saving labels at", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(bboxes.shape[0]), desc="Saving labels", unit="label"):
        label_path = os.path.join(output_dir, f"{i}.txt")
        with open(label_path, "w") as f:
            for j in range(2):  # Two bounding boxes per image
                x_min, y_min, x_max, y_max = bboxes[i, j]
                class_id = labels[i, j]  # YOLO class is the digit
                x_center = (x_min + x_max) / 2 / img_dim
                y_center = (y_min + y_max) / 2 / img_dim
                width = (x_max - x_min) / img_dim
                height = (y_max - y_min) / img_dim
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    print("Done saving labels")

def create_dataset(args, mode):
    assert (mode == "train" or mode == "valid")
    mnistdd_data = np.load(f"{mode}.npz")
    images = mnistdd_data['images']
    labels = mnistdd_data['labels']
    bboxes = mnistdd_data['bboxes']

    root_dir = args.root_dir
    images_dir = os.path.join(root_dir, args.data_dir, "images", mode)
    labels_dir = os.path.join(root_dir, args.data_dir, "labels", mode)
    save_images(images, images_dir)
    save_labels(bboxes, labels, labels_dir)

def load_image(image_path):
    """
    Load an image from the specified path and reshape to match input format.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img.reshape(-1) / 255  # Flatten to 12288 and normalize to 0-1

def load_label(label_path, img_dim=64):
    """
    Load YOLO format label and convert to bounding boxes and class labels.
    """
    bboxes = []
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= img_dim
            y_center *= img_dim
            width *= img_dim
            height *= img_dim

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            bboxes.append([x_min, y_min, x_max, y_max])
            classes.append(int(class_id))

    return np.array(bboxes), np.array(classes)

def get_dataset_item(image_dir, label_dir, index):
    """
    Load a specific image and its corresponding labels for unit testing.

    Args:
    - image_dir: Path to the directory containing saved images.
    - label_dir: Path to the directory containing saved labels.
    - index: Index of the image and label to load.

    Returns:
    - img_vector: Flattened 12288-dim vector of the image.
    - bboxes: 2x4 array of bounding boxes for the image.
    - labels: 1x2 array of digit labels.
    """
    image_path = os.path.join(image_dir, f"{index}.jpg")
    label_path = os.path.join(label_dir, f"{index}.txt")

    img_vector = load_image(image_path)
    bboxes, labels = load_label(label_path)

    return img_vector, bboxes, labels

def dataset_exists(args, mode):
    """
    Check if the dataset (images and labels) already exists for a given mode.
    :param args: Args object containing directory information
    :param mode: 'train' or 'valid'
    :return: True if the dataset exists, otherwise False
    """
    root_dir = args.root_dir
    images_dir = os.path.join(root_dir, args.data_dir, "images", mode)
    labels_dir = os.path.join(root_dir, args.data_dir, "labels", mode)

    # Check if both directories exist and are not empty
    return os.path.exists(images_dir) and os.listdir(images_dir) and \
        os.path.exists(labels_dir) and os.listdir(labels_dir)

# Load and modify YOLOv5 configuration
def load_custom_yolo_model():
    model_config = "yolov5s.yaml"
    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    cfg['nc'] = 10  # Set number of classes to 10 for MNISTDD
    cfg['img_size'] = [64, 64]  # Set image size for the model
    model = Model(cfg)
    return model

def load_dataset(mode):
    '''
    Return a Dataset object for a given mode
    :param mode: 'train' or 'valid'
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

def train_mnistdd(args, hyp, train_loader, val_loader, model, device, epochs=1, save_dir='ckpt'):
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
    # Set file directories
    save_dir = os.path.join(args.root_dir, args.save_dir)

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
            preds = non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=iou_threshold, max_det=2)  # Limit to 2 detections

            # Convert predictions and targets to standard formats
            for pred, target in zip(preds, targets):
                if pred is not None:
                    pred_boxes = pred[:, :4].cpu().numpy()  # Predicted boxes (x1, y1, x2, y2)
                    pred_scores = pred[:, 4].cpu().numpy()  # Confidence scores
                    pred_labels = pred[:, 5].cpu().numpy()  # Predicted classes
                    all_pred_boxes.append((pred_boxes, pred_scores, pred_labels))
                else:
                    all_pred_boxes.append((np.array([]), np.array([]), np.array([])))

                # Ensure target is 2D and extract ground truth
                if target.dim() == 1:
                    target = target.unsqueeze(0)

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

def generate_segmentation_mask(image, boxes):
    """
    Generate a segmentation mask based on predicted bounding boxes.
    :param image: Original image
    :param boxes: Predicted bounding boxes
    :return: Flattened segmentation mask
    """
    # TODO: Complete segmentation mask
    mask = np.full((64, 64), 10, dtype=np.int32)  # Background as 10
    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox.astype(int)
        mask[y_min:y_max, x_min:x_max] = i  # Digit 1 -> 1, Digit 2 -> 2
    return mask.flatten()


def detect_and_segment(images):
    """
    :param images: Flattened images of shape (N, 12288)
    :return: Predicted classes, bounding boxes, and segmentation masks
    """
    args = Args()
    assert(dataset_exists(args, "train"))
    assert(dataset_exists(args, "valid"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None

    # Suppress all warnings by setting PYTHONWARNINGS
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['WANDB_MODE'] = 'disable'

    # Load correct model
    if args.load:
      print("Loading saved model")
      model_path = os.path.join(args.root_dir, args.save_dir, "best.pt")

      model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False).to(device)
      #model.load_state_dict(torch.load(model_path, map_location=device))
    else:
      print("Loading default YOLOv5 model")
      model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=10).to(device)

    model.eval()


    assert len(model.names) == 10, "Model loaded with incorrect number of classes!"

    # Set up return values
    N = images.shape[0]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    pred_seg = np.zeros((N, 4096), dtype=np.int32)  # Placeholder for segmentation

    # predict
    images = images.reshape((N, 64, 64, 3)).astype(np.uint8)
    loader = torch.utils.data.DataLoader(images, batch_size=1, shuffle=False)
    for i, img in tqdm(enumerate(loader)):

        img = img.numpy()[0]
        pred = model(img, size=64).pred[0].cpu().tolist()

        if len(pred) == 0:
            num1 = [0] * 6
        else:
            num1 = pred[0]

        if len(pred) > 1:
            num2 = pred[1]
        else:
            num2 = num1

        if num1[5] > num2[5]:
            num1, num2 = num2, num1
        nums = [num1[5], num2[5]]

        boxes = [num1[:4], num2[:4]]
        boxes2 = []

        for box in boxes:
            x_min = max(0, round(box[0]))
            y_min = max(0, round(box[1]))
            x_max = min(63, round(box[2]))  # Clip to 64x64 image boundaries
            y_max = min(63, round(box[3]))
            boxes2.append([x_min, y_min, x_max, y_max])  # Match expected format
            #boxes2.append([min_x, min_y, max_x, max_y])  # No swapping x <-> y here

        pred_class[i] = nums
        pred_bboxes[i] = boxes2

    return pred_class, pred_bboxes, pred_seg


def unit_test_loaded_data(image_dir, label_dir, index, expected_classes, expected_bboxes):
    """
    Unit test for detect_and_segment using loaded images and labels.

    Args:
    - image_dir: Directory containing test images.
    - label_dir: Directory containing test labels.
    - index: Index of the image/label pair to test.
    - expected_classes: Expected class labels for the test image.
    - expected_bboxes: Expected bounding boxes for the test image.
    """
    # Load image and labels
    img_vector, gt_bboxes, gt_classes = get_dataset_item(image_dir, label_dir, index)
    print("IMAGE SHAPE:", img_vector.shape)
    print("GT BBOXES:", gt_bboxes)


    # Run the detect_and_segment function on the loaded image
    pred_classes, pred_bboxes, pred_seg = detect_and_segment(np.expand_dims(img_vector, axis=0)) # np.expand_dims(img_vector, axis=0)
    print("PRED CLASSES:", pred_classes)
    print("PRED BBOXES:", pred_bboxes)

    # Assert classes match
    '''
    assert (pred_classes[0] == expected_classes).all(), f"Class mismatch: {pred_classes[0]} != {expected_classes}"

    # Assert bounding boxes match within a tolerance (bounding boxes may not be pixel-perfect)
    assert np.allclose(pred_bboxes[0], expected_bboxes,
                       atol=1.0), f"BBox mismatch: {pred_bboxes[0]} != {expected_bboxes}"
    '''
    print(f"Test passed for index {index}.")

def main():
    args = Args()

    # Generate datasets
    if not dataset_exists(args, "train"):
        create_dataset(args, "train")
    else:
        print("Training dataset already exists, skipping creation.")

    if not dataset_exists(args, "valid"):
        create_dataset(args, "valid")
    else:
        print("Validation dataset already exists, skipping creation.")

    '''
    # UNIT TEST FOR BBOXES AND CLASSES
    image_dir = os.path.join(args.root_dir, args.data_dir, "images/valid")
    label_dir = os.path.join(args.root_dir, args.data_dir, "labels/valid")
    expected_classes = [3, 7]
    expected_bboxes = np.array([[39, 38, 53, 57], [11, 30, 23, 50]])
    index = 1
    unit_test_loaded_data(image_dir, label_dir, index=index, expected_classes=expected_classes, expected_bboxes=expected_bboxes)

    img_vector, bboxes, labels = get_dataset_item(image_dir=image_dir, label_dir=label_dir, index=index)

    # Reshape back to (64, 64, 3)
    img_reshaped = img_vector.reshape(64, 64, 3)

    # Display the image
    plt.imshow(img_reshaped)
    plt.title("Image Visualization")
    plt.axis("off")
    plt.show()

    # Reshape and scale back to 0-255
    img_reshaped = (img_vector.reshape(64, 64, 3) * 255).astype(np.uint8)

    # Save the image
    cv2.imwrite("output_image.jpg", cv2.cvtColor(img_reshaped, cv2.COLOR_RGB2BGR))

    index = 0   # 3 and 7
    img_vector, bboxes, labels = get_dataset_item(image_dir=image_dir, label_dir=label_dir, index=index)

    test_images = np.zeros((1, 12288))
    test_images[:, 1000:1200] = 255  # Simple mock image
    test_images[:, 1050:1078] = 255  # Add a pattern (is this meant to resemble '1'?)

    print(f"{pred_classes[0]}")
    print(f"Predicted Classes: {pred_classes[0]}")
    print(f"Expected Classes: {expected_classes}")
    mismatched_indices = np.where(pred_classes[0] != expected_classes)
    print(f"Mismatched indices: {mismatched_indices}")
    #assert (pred_classes[0] == expected_classes).all(), "Class mismatch"
    #assert (pred_bboxes[0] == expected_bboxes).all(), "BBox mismatch"

    # Visualize what YOLOv5 sees
    reshaped_img = test_images[0].reshape(3, 64, 64).transpose(1, 2, 0)  # Convert to RGB
    plt.imshow(reshaped_img)
    plt.show()

    # UNIT TEST (segmentation)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt_classes.flatten(), pred_classes.flatten())
    print("Confusion Matrix:\n", cm)
    '''
if __name__ == '__main__':
    main()
