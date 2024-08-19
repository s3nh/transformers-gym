import torch
from transformers import AutoModelForCausalLM


def check_matching_weights(model0, model1):
    # Just check if there are mismatched model sizes
    model0 = AutoModelForCausalLM.from_pretrained(model0, torch_dtype = torch.bfloat16).cpu()
    model1 = AutoModelForCausalLM.from_pretrained(model1, torch_dtype = torch.bfloat16).cpu()
    mismatch_diffs = []
    any_mismatch = False
    for (name0, param0), (name1, param1) in zip(model0.named_parameters(), model1.named_parameters()):
        if not (param0.data == param1.data).all():
            any_mismatch = True
            diff = torch.sum(torch.abs(param0.data - param1.data)).item()
            mismatch_diffs.append(diff)
            print("Mismatched weights", name0, name1, diff)
    if not any_mismatch:
        print("No mismatched weights")
    else:
        print("Mean abs mismatched", np.mean(mismatch_diffs), "Std abs mismatched", np.std(mismatch_diffs))
    del model0, model1; gc.collect(); torch.cuda.empty_cache()


def load_model(model_id):
    """Load a model from Hugging Face Hub."""
    return AutoModel.from_pretrained(model_id, device_map = 'cpu')

def are_models_near_duplicates(model_id_1, model_id_2, tolerance=1e-4):
    """
    Compare two models to check if they are near duplicates.

    Parameters:
    model_id_1 (str): Model ID for the first model.
    model_id_2 (str): Model ID for the second model.
    tolerance (float): Tolerance level for parameter comparison.

    Returns:
    bool: True if models are near duplicates, False otherwise.
    """
    model1 = load_model(model_id_1)
    model2 = load_model(model_id_2)

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(param1, param2, atol=tolerance):
            return False
    return True
