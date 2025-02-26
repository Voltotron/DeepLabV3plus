import torch

def pix_acc(target, outputs, num_classes):
    """
    Calculates pixel accuracy, given target and output tensors 
    and number of classes.
    """
    labeled = (target > 0) * (target <= num_classes)
    _, preds = torch.max(outputs.data, 1)
    correct = ((preds == target) * labeled).sum().item()
    return labeled, correct

def iou(preds, target, class_index):
    """
    Calculates Intersection over Union (IoU) for a specific class.
    :param preds: Model predictions (output tensor).
    :param target: Ground truth labels (target tensor).
    :param class_index: Index of the class to compute IoU for (e.g., 1 for concrete).
    """
    preds = torch.argmax(preds, dim=1)  # Get predicted class indices
    pred_inds = preds == class_index
    target_inds = target == class_index

    intersection = (pred_inds & target_inds).sum().float()
    union = (pred_inds | target_inds).sum().float()

    if union == 0:
        return float('nan')  # Avoid division by zero if the class is not present
    else:
        return (intersection / union).item()