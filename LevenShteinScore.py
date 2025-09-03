import json
import os

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def load_ground_truth(labels_file):
    """Load ground truth labels from labels.json"""
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    # Create dictionary for quick lookup
    gt_dict = {}
    for item in labels_data:
        gt_dict[item['image_id']] = item['captcha_string']
    
    return gt_dict

def calculate_levenshtein_for_threshold(predictions_file, ground_truth_dict):
    """Calculate average Levenshtein distance for a threshold"""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    total_distance = 0
    total_images = 0
    
    for pred in predictions:
        image_id = pred['image_id']
        pred_string = pred['captcha_string']
        
        if image_id in ground_truth_dict:
            gt_string = ground_truth_dict[image_id]
            distance = levenshtein_distance(pred_string, gt_string)
            total_distance += distance
            total_images += 1
    
    avg_distance = total_distance / total_images if total_images > 0 else 0
    return avg_distance

def compare_thresholds():
    """Compare Levenshtein distances for different thresholds"""
    labels_file = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val/labels.json"
    thresholds = [0.3, 0.35, 0.4, 0.45]
    
    print("🚀 Calculating Levenshtein Distance for Different Thresholds")
    print("="*60)
    
    # Load ground truth
    ground_truth = load_ground_truth(labels_file)
    
    for thresh in thresholds:
        predictions_file = f"predictions_conf_{thresh:.2f}.json"
        
        if os.path.exists(predictions_file):
            avg_distance = calculate_levenshtein_for_threshold(predictions_file, ground_truth)
            print(f"Threshold {thresh:.2f}: Average Levenshtein Distance = {avg_distance:.3f}")
        else:
            print(f"Threshold {thresh:.2f}: File not found")

# Add this to your script temporarily
def check_data_counts():
    labels_file = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val/labels.json"
    
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    print(f"Ground truth has: {len(labels_data)} images")
    
    # Check one prediction file
    if os.path.exists("predictions_conf_0.35.json"):
        with open("predictions_conf_0.35.json", 'r') as f:
            pred_data = json.load(f)
        print(f"Predictions have: {len(pred_data)} images")

if __name__ == "__main__":
    compare_thresholds()
    check_data_counts()