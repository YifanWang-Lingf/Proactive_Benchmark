from . import (
    llava_ov,
    internvl_2d5,
    qwen2_5_vl_7b,
    longva_7b,
    internlm_xcomposer_2_5,
    openai_api
)


def load_model(model_type, *args,  **kwargs):
    if model_type == "llava_ov":
        return llava_ov.load_model(*args, **kwargs)
    if model_type == "internvl_2d5":
        return internvl_2d5.load_model(*args, **kwargs)
    if model_type == "openai_api":
        return openai_api.load_model(*args, **kwargs)
    
    if model_type == "qwen2_5_vl_7b":
        return qwen2_5_vl_7b.load_model(*args, **kwargs)
    if model_type == "longva_7b":
        return longva_7b.load_model(*args, **kwargs)
    if model_type == "internlm_xcomposer_2.5":
        return internlm_xcomposer_2_5.load_model(*args, **kwargs)  
    
    raise NotImplementedError(f"Model {model_type} not implemented")


def inference(model_type, *args, **kwargs):
    if model_type == "llava_ov":
        return llava_ov.inference(*args, **kwargs)
    if model_type == "internvl_2d5":
        return internvl_2d5.inference(*args, **kwargs)
    if model_type == "openai_api":
        return openai_api.inference(*args, **kwargs)
    if model_type == "qwen2_5_vl_7b":
        return qwen2_5_vl_7b.inference(*args, **kwargs)
    if model_type == "longva_7b":   
        return longva_7b.inference(*args, **kwargs)
    if model_type == "internlm_xcomposer_2.5":
        return internlm_xcomposer_2_5.inference(*args, **kwargs)
    raise NotImplementedError(f"Model {model_type} not implemented")
