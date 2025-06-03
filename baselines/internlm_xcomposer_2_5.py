import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from utils.video_utils import load_video

def process_vision_info(messages, processor, config=None):
    """
    将 messages 中的视频路径转换为模型可用的视觉输入张量 (vision_inputs, video_inputs)。
    
    参数：
      - messages: List[dict]，每个 dict 形如：
          {
            "role": "user" or "assistant",
            "content": [
              {"type": "video", "video": "<视频文件路径>"},
              {"type": "text",  "text": "<相关文本>"}
            ]
          }
        这里只取第一个 message 里的第一个 "type":"video" 条目作为视频输入。
      - processor: InternLM-XComposer-2.5 的 AutoProcessor 实例，用于将 PIL.Image 转为张量。
      - config: 模型的 config 对象（如果需要传入以指导图像预处理）。
    
    返回：
      - vision_inputs: torch.Tensor，形状 [batch_size, 3, H, W]（batch_size=1）
      - video_inputs: torch.Tensor or None（如果模型需要视频序列张量，可以在此处返回；若不需要，则返回 None）
    """
    # 1. 从 messages 中提取视频文件路径
    video_path = None
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "video":
                video_path = item.get("video")
                break
        if video_path:
            break

    if video_path is None:
        # 没有找到视频路径，直接返回 (None, None)
        return None, None

    # 2. 使用 load_video 抽帧：得到 List[np.ndarray]，每个元素形状 (3, H, W)
    #    这里假设如果没有显式传递起止时间，则默认取整个视频
    pixel_list = load_video(video_path)
    pil_frames = []
    if pixel_list:
        for frame_np in pixel_list:
            # 将 (3, H, W) 转为 (H, W, 3) 后转换为 PIL.Image
            pil_frames.append(Image.fromarray(frame_np.transpose(1, 2, 0)))
    # 抽帧完成后立即删除中间 list，释放 CPU 内存
    del pixel_list

    if not pil_frames:
        return None, None

    # 3. 调用 processor 处理 PIL.Image 列表，得到 vision_inputs
    #    internlm 的 AutoProcessor 在处理图像时会返回一个 dict，其中包含 'pixel_values'
    vision_data = processor(
        images=pil_frames,
        return_tensors="pt",
        add_special_tokens=False
    )
    # vision_inputs 形状 [batch_size, 3, H, W]，如果有多帧，batch_size 会等于帧数
    vision_inputs = vision_data["pixel_values"]

    # 4. 对于多模态模型，如果需要将视频序列视为额外输入，可以在这里组装 video_inputs
    #    如果不需要，可以直接设为 None。
    video_inputs = None

    return vision_inputs, video_inputs

def clean_response(raw_text: str) -> str:
    """
    只保留 raw_text 中“assistant\n”之后的内容（去掉 system/user/assistant 标签及前面的部分）。
    """
    lower = raw_text.lower()
    tag = "assistant\n"
    idx = lower.rfind(tag)
    if idx >= 0:
        return raw_text[idx + len(tag):].strip()
    else:
        return raw_text.strip()

def load_model(llm_pretrained, attn_implementation):
    """
    与调用方保持完全一致的签名：load_model(llm_pretrained, attn_implementation)

    这里只需使用 llm_pretrained（如 "internlm/internlm-xcomposer2d5-7b" 或 "internlm/internlm-xcomposer2d5-7b-chat"），
    attn_implementation 在 InternLM-XComposer-2.5 中不生效（模型构造不接受 flash-attn 参数），
    故不启用 device_map="auto"（注释里标明不使用它）。

    返回：
      {
        "model": <InternLM-XComposer2.5 的 AutoModelForCausalLM 实例>,
        "tokenizer": <AutoTokenizer 实例>,
        "processor": <AutoProcessor 实例>,
        "generation_config": {"max_new_tokens": 256, "do_sample": True}
      }
    """

    # 1. 加载 Tokenizer 和模型，使用 bfloat16 节省显存
    #    “trust_remote_code=True” 保证加载 InternLM-XComposer 的自定义代码:contentReference[oaicite:0]{index=0}
    tokenizer = AutoTokenizer.from_pretrained(
        llm_pretrained,
        trust_remote_code=True,
        use_fast=False  # 有些 InternLM 版本的 processor 需要慢速 tokenization
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_pretrained,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
        # device_map="auto"  # 注：此处不启用 device_map="auto"
    ).eval().cuda()

    # 2. 加载 AutoProcessor（会包含 InternLM-XComposer 需要的视觉预处理逻辑）:contentReference[oaicite:1]{index=1}
    processor = AutoProcessor.from_pretrained(
        llm_pretrained,
        trust_remote_code=True
    )

    # 3. 保留生成配置，与之前 Qwen/LongVA 保持一致
    generation_config = {"max_new_tokens": 256, "do_sample": True}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "generation_config": generation_config
    }

def inference(
    baseline_model,
    video_data,
    text_data,
    need_judge_answerable=True,
    debug_print=False
):
    """
    InternLM-XComposer-2.5 推理：
      • 视频帧 → process_vision_info → (vision_inputs, video_inputs)
      • processor(text=..., vision_inputs=..., videos=...) 打包
      • 可答性判定严格取最后一个 token
      • 正式回答后用 clean_response()
    """
    device    = "cuda"
    model     = baseline_model["model"]
    tokenizer = baseline_model["tokenizer"]
    processor = baseline_model["processor"]
    gen_cfg   = baseline_model["generation_config"]

    # -------- 文本变量 --------
    original_q  = text_data["original_question"]
    add_text    = text_data.get("additional_text_input", "").strip()
    prev_output = text_data.get("previous_turns_output", "").strip()

    # -------- 1. 可答性判定 --------
    if need_judge_answerable:
        prompt_yesno = (
            (add_text + "\n") if add_text else ""
        ) + "Is the following question answerable by the video? Only reply with Yes or No.\n" + original_q

        chatml_yesno = (
            "<|im_start|>system\nYou are a helpful assistant. Reply with Yes or No only.\n<|im_end|>\n"
            "<|im_start|>user\n<video>\n" + prompt_yesno + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 1.1 视觉预处理：messages → (vision_inputs, video_inputs)
        messages_yesno = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_data["video_file"]},
                {"type": "text",  "text": prompt_yesno}
            ]
        }]
        vision_inputs_yesno, video_inputs_yesno = process_vision_info(messages_yesno, processor)

        if vision_inputs_yesno is not None:
            vision_inputs_yesno = vision_inputs_yesno.to(device, dtype=torch.bfloat16)
        if video_inputs_yesno is not None:
            video_inputs_yesno = video_inputs_yesno.to(device, dtype=torch.bfloat16)

        inputs_yesno = processor(
            text=[chatml_yesno],
            vision_inputs=vision_inputs_yesno,
            videos=video_inputs_yesno,
            fps=video_data["frame_fps"],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            gen_yesno = model.generate(
                **inputs_yesno,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample     =gen_cfg["do_sample"]
            )
        raw_yesno = tokenizer.batch_decode(gen_yesno, skip_special_tokens=True)[0]

        if debug_print:
            print("--- 模型判断是否当前是否具有可回答性：模型的原始输出内容 ---\n", raw_yesno)

        # 1.2 取最后一行最后单词
        last_line  = raw_yesno.splitlines()[-1].strip() if raw_yesno.splitlines() else raw_yesno.strip()
        last_token = last_line.split()[-1].lower() if last_line.split() else raw_yesno.strip().lower()
        answerable = "Yes" if last_token == "yes" else ("No" if last_token == "no" else raw_yesno.strip())

        if debug_print:
            print("--- 模型判断是否当前是否具有可回答性：模型处理后的输出内容 ---\n", answerable)

        # --------- 显存优化1 释放掉第一轮可回答性推理的中间数据张量   ---------
        del inputs_yesno, gen_yesno, raw_yesno
        del vision_inputs_yesno, video_inputs_yesno
        torch.cuda.empty_cache()
        # ------------------------------------------------------------------
        
        
        
    else:
        answerable = "Yes"

    # -------- 2. 正式回答 --------
    if "yes" in answerable.lower():
        if prev_output:
            prompt_final = f"{prev_output}\n{add_text}\n{original_q}" if add_text else f"{prev_output}\n{original_q}"
        else:
            prompt_final = f"{add_text}\n{original_q}" if add_text else original_q

        chatml_final = (
            "<|im_start|>system\nYou are a helpful assistant. Provide a concise answer based on the video.\n<|im_end|>\n"
            "<|im_start|>user\n<video>\n" + prompt_final + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 2.1 视觉预处理
        messages_final = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_data["video_file"]},
                {"type": "text",  "text": prompt_final}
            ]
        }]
        vision_inputs_final, video_inputs_final = process_vision_info(messages_final, processor)
        if vision_inputs_final is not None:
            vision_inputs_final = vision_inputs_final.to(device, dtype=torch.bfloat16)
        if video_inputs_final is not None:
            video_inputs_final  = video_inputs_final.to(device, dtype=torch.bfloat16)

        inputs_final = processor(
            text=[chatml_final],
            vision_inputs=vision_inputs_final,
            videos=video_inputs_final,
            fps=video_data["frame_fps"],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            gen_final = model.generate(
                **inputs_final,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample     =gen_cfg["do_sample"]
            )
        raw_final = tokenizer.batch_decode(gen_final, skip_special_tokens=True)[0]

        if debug_print:
            print("--- 模型正式回答：模型的原始输出内容 ---\n", raw_final)

        response = clean_response(raw_final)

        if debug_print:
            print("--- 模型正式：模型处理后的输出内容 ---\n", response)

        # --------- 显存优化2 释放掉第一轮可回答性推理的中间数据张量   ---------
        del inputs_final, gen_final, raw_final
        del vision_inputs_final, video_inputs_final
        torch.cuda.empty_cache()
        # ------------------------------------------------------------------
        
    else:
        response = None

    return {"answerable": answerable, "response": response}

