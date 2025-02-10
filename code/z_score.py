import numpy as np

org = np.array(['...']) 
adv = np.array(['...'])  

# Function to calculate Z-score
def z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev

# Function to compute detection rate and false positive rate
def evaluate_threshold(org, adv, threshold):
    # Calculate Z-scores for org and adv data
    org_z_scores = z_score(org)
    adv_z_scores = z_score(adv)
    
    true_positives = np.sum(np.abs(adv_z_scores) > threshold)
    false_positives = np.sum(np.abs(org_z_scores) > threshold)
    detection_rate = true_positives / len(adv)
    false_positive_rate = false_positives / len(org)
    
    return detection_rate, false_positive_rate

# Function to find the threshold based on Z-scores of org
def find_threshold_using_zscore(org, adv, method='max', percentile=95):
    # Calculate Z-scores for org data
    org_z_scores = z_score(org)
    
    # Choose threshold based on method
    if method == 'max':
        threshold = np.max(np.abs(org_z_scores))  # Use max absolute Z-score
    elif method == 'percentile':
        threshold = np.percentile(np.abs(org_z_scores), percentile)  # Use specified percentile
    
    # Calculate detection rate and false positive rate using this threshold
    detection_rate, false_positive_rate = evaluate_threshold(org, adv, threshold)
    
    return threshold, detection_rate, false_positive_rate



threshold, detection_rate, fpr = find_threshold_using_zscore(org, adv, method='max')

print(f"Threshold: {threshold}")
print(f"Detection Rate: {detection_rate:.4f}")
print(f"False Positive Rate: {fpr:.4f}")