# Fully chatGPT generated code to compute and save SDP results with parameter hashing

import numpy as np
import hashlib
import json
import base64
from sdp_solver import solve_sdp

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # Encode arrays as base64 strings for compactness
            return {
                '__ndarray__': base64.b64encode(obj.tobytes()).decode('utf-8'),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        return super().default(obj)

def numpy_object_hook(dct):
    """Custom object hook for decoding NumPy arrays"""
    if '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def dict_hash(dictionary):
    """MD5 hash of a dictionary with NumPy support"""
    dhash = hashlib.md5()
    
    # Serialize with custom encoder
    encoded = json.dumps(
        dictionary, 
        sort_keys=True, 
        cls=NumpyEncoder
    ).encode('utf-8')
    
    dhash.update(encoded)
    return dhash.hexdigest()

def compute_and_save_sdp_results(params, filename="sdp_results.npz"):
    """Compute SDP results and save to file with parameter hash"""
    # Compute hash of parameters
    param_hash = dict_hash(params)
    
    # Solve SDP
    sdp_result = solve_sdp(
        params["A"], params["B"], params["C"],
        params["X0_hat"], params["W_hat"], params["V_hat"],
        params["dt"], params["T"],
        params["rho_x0"], params["rho_w"], params["rho_v"]
    )
    
    # Add hash to params for saving
    params_with_hash = params.copy()
    params_with_hash["hash"] = param_hash
    
    # Save results
    np.savez(
        filename,
        params=params_with_hash,
        X0_robust=sdp_result["X0"],
        W0_robust=sdp_result["W_list"],
        V0_robust=sdp_result["V_list"],
        status=sdp_result["status"],
        optimal_value=sdp_result["optimal_value"]
    )
    print(f"SDP results saved to {filename} with hash {param_hash}")
    return sdp_result

def load_sdp_results(filename="sdp_results.npz"):
    """Load precomputed SDP results from file"""
    data = np.load(filename, allow_pickle=True)
    loaded = {
        "params": data["params"].item(),
        "X0_robust": data["X0_robust"],
        "W_list_robust": data["W0_robust"],
        "V_list_robust": data["V0_robust"],
        "status": data["status"].item(),
        "optimal_value": data["optimal_value"].item()
    }
    return loaded

def should_recompute_sdp(current_params, filename="sdp_results.npz"):
    """Check if SDP needs recomputation based on parameter changes"""
    try:
        # Try to load existing results
        saved_data = load_sdp_results(filename)
        saved_params = saved_data["params"]
        
        # Compare current parameters with saved ones
        current_hash = dict_hash(current_params)
        saved_hash = saved_params.get("hash", "")
        
        return current_hash != saved_hash
    except FileNotFoundError:
        return True  # File doesn't exist
    except KeyError as e:
        print(f"Missing key in SDP file: {e}")
        return True  # File is missing required data
    except Exception as e:
        print(f"Error loading SDP file: {str(e)}")
        return True  # Other errors