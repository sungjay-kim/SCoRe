import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create symlinks for LatentCodes mapping Index -> InstanceName")
    parser.add_argument("--specs", required=True, help="Path to DeepSDF specs.json")
    parser.add_argument("--latent_dir", required=True, help="Path to LatentCodes directory (containing 0.pth, 1.pth...)")
    parser.add_argument("--out_dir", required=True, help="Output directory for symlinks")
    args = parser.parse_args()

    # Load Specs to find Split Path
    with open(args.specs, 'r') as f:
        specs = json.load(f)
    
    train_split_path = specs.get("TrainSplit")
    test_split_path = specs.get("TestSplit") # DeepSDF usually loads Train THEN Test to form the full dataset indices?
    
    # DeepSDF Training Logic (replicated):
    # It loads TrainSplit, then appends TestSplit (if Validation/Test logic is integrated).
    # Wait, DeepSDF usually ONLY optimizes latent codes for the TRAIN split during training.
    # Test codes are optimized at inference time.
    # HOWEVER, kmelon_latent32 has 1518 codes. 
    # Let's count the Train Split size.
    
    with open(train_split_path, 'r') as f:
        train_split = json.load(f)
    
    # DeepSDF data.py flattening order:
    # keys are sorted? No, standard DeepSDF iterates dictionary keys. 
    # In Py3.7+ insertion order is preserved.
    # We assume 'kmelon_centered' is first, then 'kmelon_augmented'.
    
    instance_names = []
    
    # Check if keys are present
    print(f"Loading instances from {train_split_path}...")
    for dataset_name in train_split:
        if dataset_name.startswith("_"): continue # skip meta
        # For each class? The structure uses Dataset -> Class -> Instances?
        # Actually our file has: { "kmelon_centered": [...], "kmelon_augmented": [...] } inside the root?
        # Let's check the structure more carefully. 
        # DeepSDF usually expects { "ShapeNetV2": { "02691156": [...] } }
        # Our key is "."? Earlier `head` showed `".": { "kmelon_centered": ... }`
        # So we iterate train_split["."]
        
        root_data = train_split.get(".", train_split) # Handle if "." wrapper exists or not
        
        for class_name in root_data:
             if class_name.startswith("_"): continue
             instances = root_data[class_name]
             print(f"  Found class {class_name}: {len(instances)} instances")
             instance_names.extend(instances)
             
    print(f"Total instances in Train Split: {len(instance_names)}")
    
    # Verify count matches LatentCodes
    latent_files = list(Path(args.latent_dir).glob("*.pth"))
    # Filter for integer-named files
    latent_indices = [int(f.stem) for f in latent_files if f.stem.isdigit()]
    max_index = max(latent_indices) if latent_indices else -1
    count_latents = len(latent_indices)
    
    print(f"LatentCodes count: {count_latents}, Max Index: {max_index}")
    
    if count_latents != len(instance_names):
        print("WARNING: Count mismatch! LatentCodes vs TrainSplit.")
        # It's possible DeepSDF trained on Test split too? Or filtered some?
        # We will try to map 0..N assuming sequential order.
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for idx, name in enumerate(instance_names):
        src = Path(args.latent_dir) / f"{idx}.pth"
        dst = Path(args.out_dir) / f"{name}.pth"
        
        if not src.exists():
            print(f"Missing source latent: {src} for {name}")
            continue
            
        if dst.exists():
            os.remove(dst)
            
        os.symlink(src, dst)
        
    print(f"Created {len(instance_names)} symlinks in {args.out_dir}")

if __name__ == "__main__":
    main()
