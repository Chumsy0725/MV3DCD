import os
import torch
import cv2
import argparse
from tqdm import tqdm
import torchmetrics

# Initialize TorchMetrics for mIoU and F1
miou_metric = torchmetrics.JaccardIndex(task='binary')
f1_metric = torchmetrics.F1Score(task='binary')

# Ensure 'plots' directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Function to calculate mIoU and F1

def calculate_metrics_torch(ground_truth, predicted):
    ground_truth_tensor = torch.tensor(ground_truth).unsqueeze(0)
    predicted_tensor = torch.tensor(predicted).unsqueeze(0)

    # mIoU
    miou = miou_metric(predicted_tensor, ground_truth_tensor).item()

    # F1 Score
    f1 = f1_metric(predicted_tensor, ground_truth_tensor).item()

    return miou, f1

# Function to find a matching predicted file by name, ignoring the extension
def find_matching_file(gt_file_name, predicted_dir):
    gt_base_name, _ = os.path.splitext(gt_file_name)
    for pred_file in os.listdir(predicted_dir):
        pred_base_name, _ = os.path.splitext(pred_file)
        if gt_base_name == pred_base_name:
            return os.path.join(predicted_dir, pred_file)
    return None

# Function to evaluate segmentation with binary masks
def evaluate_segmentation(ground_truth_dir, predicted_binary_dir):
    miou_scores = []
    f1_scores = []

    gt_files = os.listdir(ground_truth_dir)

    for gt_file_name in tqdm(gt_files, desc="Evaluating segmentation masks", unit="mask"):
        ground_truth_path = os.path.join(ground_truth_dir, gt_file_name)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        # ground_truth = cv2.resize(ground_truth, (224, 224), interpolation=cv2.INTER_NEAREST)
        _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        ground_truth = ground_truth // 255

        predicted_binary_path = find_matching_file(gt_file_name, predicted_binary_dir)

        if predicted_binary_path:
            predicted_binary = cv2.imread(predicted_binary_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth.shape != predicted_binary.shape:
                predicted_binary = cv2.resize(predicted_binary, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, predicted_binary = cv2.threshold(predicted_binary, 127, 255, cv2.THRESH_BINARY)
            predicted_binary = predicted_binary // 255

            miou, f1 = calculate_metrics_torch(ground_truth, predicted_binary)
            miou_scores.append(miou)
            f1_scores.append(f1)

    if miou_scores:
        mean_miou = torch.tensor(miou_scores).mean().item()
        mean_f1 = torch.tensor(f1_scores).mean().item()

        print(f"Mean IoU: {mean_miou}")
        print(f"Mean F1: {mean_f1}")

    return mean_miou, mean_f1

# Main function to parse arguments and run evaluation
def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results using mIoU and F1")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth masks directory")
    parser.add_argument("--pred_binary", type=str, required=True, help="Path to the binary predicted masks directory")

    args = parser.parse_args()

    # Run the evaluation
    evaluate_segmentation(args.gt, args.pred_binary)

if __name__ == "__main__":
    main()
