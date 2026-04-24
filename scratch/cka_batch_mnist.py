import subprocess
import re
import os

def run_cka(args_list):
    cmd = ["python", "playground/circuit_lab.py", "cka"] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    
    latent_match = re.search(r"Full Model Similarity \(Latent\)\s+([\d\.]+)", out)
    circuit_match = re.search(r"Average Circuit CKA\s+([\d\.]+)", out)
    circuit_class_matches = re.findall(r"Class (\d+)\s+CKA = ([\d\.]+)", out)
    latent_class_matches = re.findall(r"Class (\d+) \(Latent\)\s+([\d\.]+)", out)
    
    latent_total = latent_match.group(1) if latent_match else "N/A"
    circuit_avg = circuit_match.group(1) if circuit_match else "N/A"
    per_class_circ = {f"Circ_{m[0]}": m[1] for m in circuit_class_matches}
    per_class_lat = {f"Lat_{m[0]}": m[1] for m in latent_class_matches}
    
    return latent_total, circuit_avg, per_class_circ, per_class_lat

exp_iid = r"E:\FedMI_Saves\iid_experiment_resnet_mnist"
exp_niid = r"E:\FedMI_Saves\niid_experiment_resnet_mnist"
class_ids = ["0", "1", "2", "3", "4"]

# --- 1. Global Progression (IID vs Non-IID, Rounds 1-10) ---
print("\n--- 1. Global Progression (IID vs Non-IID) ---")
global_results = []
for r in [1, 2, 5, 8, 10]:
    lt, ca, pc, pl = run_cka([
        "--exp_a", exp_iid,
        "--exp_b", exp_niid,
        "--round", str(r),
        "--source", "global",
        "--classes", *class_ids,
        "--max_samples", "2048"
    ])
    global_results.append({"Round": r, "Latent Total": lt, "Circuit Avg": ca, **pc, **pl})

# --- 2. Intra-Experiment Progression (NIID R10 vs R50) ---
print("\n--- 2. Intra-Experiment Progression (NIID R10 vs R50) ---")
niid_long_results = []
lt, ca, pc, pl = run_cka([
    "--exp_a", exp_niid,
    "--exp_b", exp_niid,
    "--round_a", "10",
    "--round_b", "50",
    "--source", "global",
    "--classes", *class_ids,
    "--max_samples", "2048"
])
niid_long_results.append({"Comparison": "NIID R10 vs R50", "Latent Total": lt, "Circuit Avg": ca, **pc, **pl})

# --- 3. Inter-Client Comparison (Non-IID, Round 50) ---
print("\n--- 3. Inter-Client Comparison (Non-IID, Round 50) ---")
client_results = []
for c_a, c_b in [(0, 1), (0, 2), (1, 2)]:
    lt, ca, pc, pl = run_cka([
        "--exp_a", exp_niid,
        "--exp_b", exp_niid,
        "--round", "50",
        "--client_a", str(c_a),
        "--client_b", str(c_b),
        "--source", "local",
        "--classes", *class_ids,
        "--max_samples", "2048"
    ])
    client_results.append({"Comparison": f"C{c_a} vs C{c_b}", "Latent Total": lt, "Circuit Avg": ca, **pc, **pl})

# --- Formatting ---
def print_detailed_table(results, title, header_key):
    print(f"\n### {title}")
    circ_headers = sorted([k for k in results[0].keys() if k.startswith("Circ_")])
    lat_headers = sorted([k for k in results[0].keys() if k.startswith("Lat_")])
    headers = [header_key, "Latent Total", "Circuit Avg"] + lat_headers + circ_headers
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in results:
        cells = [str(row.get(h, "N/A")) for h in headers]
        print("| " + " | ".join(cells) + " |")

print("\n" + "="*80 + "\nRESULTS: MNIST RESNET LONG-HORIZON STUDY\n" + "="*80)
print_detailed_table(global_results, "Global Progression (IID vs Non-IID)", "Round")
print_detailed_table(niid_long_results, "Non-IID Long-Horizon Convergence", "Comparison")
print_detailed_table(client_results, "Inter-Client Similarity (NIID R50)", "Comparison")
