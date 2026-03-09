import os
import sys
import numpy as np
import pandas as pd
import torch
import itertools
from dialz import SteeringVector
import seaborn as sns
import matplotlib.pyplot as plt

# Default to mistral
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
if len(sys.argv) > 1:
    model_name = sys.argv[1]

model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}
model_short_name = model_short_names.get(model_name)

# All axes
bbq_axes = ["age", "appearance", "disability", "gender", "nationality", "race", "religion", "socioeconomic"]
sycophancy_axes = ["sycophancy"]
all_axes = bbq_axes + sycophancy_axes

vector_types = ["train", "train+prompt", "generate_ss", "generate_qa"]

# We'll save the similarity scores layer-by-layer
out_dir = f"../data/vector_similarities/{model_short_name}"
os.makedirs(out_dir, exist_ok=True)
figs_dir = f"../figs/{model_short_name}/vector_similarities"
os.makedirs(figs_dir, exist_ok=True)

for v_type in vector_types:
    print(f"Loading vectors for {v_type}...")
    
    # Load all vectors for this v_type
    loaded_vectors = {}
    for axis in all_axes:
        # Load from GGUF
        path = f"../vectors/{model_short_name}/{v_type}/{axis}.gguf"
        if os.path.exists(path):
            try:
                loaded_vectors[axis] = SteeringVector.import_gguf(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: {path} not found.")
            
    if len(loaded_vectors) < 2:
        print(f"Skipping {v_type} because not enough vectors were found.")
        continue
        
    # Find all shared layers
    # loaded_vectors[axis].directions is usually a dict {layer: torch.Tensor}
    # We find the intersection of all layer keys
    shared_layers = list(loaded_vectors.values())[0].directions.keys()
    for axis, vec in loaded_vectors.items():
        shared_layers = set(shared_layers) & set(vec.directions.keys())
    
    shared_layers = sorted(list(shared_layers))
    
    if not shared_layers:
        print(f"No shared layers found for {v_type} vectors.")
        continue

    # Create similarity comparison per layer
    all_results = []
    
    axis_names = list(loaded_vectors.keys())
    
    for layer in shared_layers:
        # Get tensors for all axes at this layer, normalize them
        vectors_at_layer = {}
        for axis in axis_names:
            arr = loaded_vectors[axis].directions[layer]
            t = torch.tensor(arr).float()
            # It should be a 1D tensor
            if t.dim() > 1:
                t = t.squeeze()
            vectors_at_layer[axis] = t / t.norm(p=2)
            
        # Compute pairs
        for ax1, ax2 in itertools.combinations(axis_names, 2):
            v1 = vectors_at_layer[ax1]
            v2 = vectors_at_layer[ax2]
            
            cos_sim = torch.dot(v1, v2).item()
            all_results.append({
                "layer": layer,
                "vector_type": v_type,
                "axis_1": ax1,
                "axis_2": ax2,
                "cosine_similarity": cos_sim
            })
            
    # Save raw data
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, f"{v_type}_cosine_similarities.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved similarities for {v_type} to {csv_path}")
    
    # Let's optionally plot heatmaps for each layer, but that might be 32 x 4 = 128 heatmaps.
    # We can plot a line graph: Average cosine similarity vs layer
    # Or similarity of everything to Sycophancy across layers
    if "sycophancy" in axis_names:
        plt.figure(figsize=(10, 6))
        
        # Plot similarity of each bbq axis to sycophancy
        for bbq_axis in [a for a in axis_names if a != "sycophancy"]:
            # Filter df
            sub_df = df[((df["axis_1"] == "sycophancy") & (df["axis_2"] == bbq_axis)) | 
                        ((df["axis_2"] == "sycophancy") & (df["axis_1"] == bbq_axis))]
            
            # Sort by layer
            sub_df = sub_df.sort_values(by="layer")
            if not sub_df.empty:
                plt.plot(sub_df["layer"], sub_df["cosine_similarity"], marker='o', label=f"syco vs {bbq_axis}")
                
        plt.title(f"Cosine Similarity to Sycophancy Vector per Layer ({v_type})")
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"syco_similarity_{v_type}.png"))
        plt.close()

print("Done computing vector similarities!")
