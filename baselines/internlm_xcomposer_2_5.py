import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from utils.video_utils import load_video
import torch.nn.functional as F
import os
import numpy as np
import cv2

def load_model(llm_pretrained, attn_implementation):
    """
    这里只需使用 llm_pretrained（如 "internlm/internlm-xcomposer2d5-7b" 或 "internlm/internlm-xcomposer2d5-7b-chat"），
    attn_implementation 在 InternLM-XComposer-2.5 中不生效（模型构造不接受 flash-attn 参数），

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
    ).eval().cuda()
    model.tokenizer = tokenizer
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
    
def frames_to_mp4_opencv(np_frames, fps, out_path):
    """
    将 np_frames（list of 3×H×W uint8 numpy）写成 out_path 的 MP4。
    """
    # 将第一帧转成 H, W
    C, H, W = np_frames[0].shape
    # OpenCV 要 BGR，帧尺寸是 (W, H)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 "avc1"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), True)

    for f in np_frames:
        # f: (C,H,W) → (H,W,C) → BGR
        img = cv2.cvtColor(f.transpose(1,2,0), cv2.COLOR_RGB2BGR)
        writer.write(img)
    writer.release()
    return out_path

def inference(
    baseline_model,
    video_data,
    text_data,
    need_judge_answerable=True,
    debug_print=False
):

    device    = "cuda"
    model     = baseline_model["model"]
    tokenizer = baseline_model["tokenizer"]
    processor = baseline_model["processor"]
    gen_cfg   = baseline_model["generation_config"]

    # -------- 文本变量 --------
    original_q  = text_data["original_question"]
    add_text    = text_data.get("additional_text_input", "").strip()
    prev_output = text_data.get("previous_turns_output", "").strip()

    # -------- 抽帧，只抽取一个chunk也即是start_sec ~ end_sec之间的视频 --------
    np_frames = load_video(
        video_file=video_data["video_file"],
        start_sec=video_data["start_sec"],
        end_sec  =video_data["end_sec"],
        output_resolution=video_data["frame_resolution"],
        output_fps=video_data["frame_fps"]
    )
    chunk_path = "/share2/wyf/projects/proactive_benchmark/frames/chunk.mp4"
    frames_to_mp4_opencv(np_frames, video_data["frame_fps"], chunk_path)
    
    # import debugpy
    # debugpy.breakpoint()  # 调试用
    
    
    # SAVE_DIR = "/share2/wyf/projects/proactive_benchmark/frames"; 
    # os.makedirs(SAVE_DIR, exist_ok=True)
    # frame_paths = []

    # for i, f in enumerate(np_frames):    # f: 3×H×W, uint8
    #     arr = np.transpose(f, (1, 2, 0))            # → H×W×3
    #     path = f"{SAVE_DIR}/frame_{i:04d}.jpg"
    #     Image.fromarray(arr, mode="RGB").save(path, quality=95)
    #     frame_paths.append(path)
    
    # !!!!!!!!!!!!!!!!!! 注意 !!!!!!!!!!!!!!!!!!
    # 经过测试与查询，xcomposer系列模型总是倾向于先进行一下图文内容描述
    # 所以在可答性判定时，最好限定死模型的输出格式为 "Yes" 或 "No"，而不是其他内容
    # 因此可达性判断时用另一套生成配置，ax_new_tokens=2限定输出token数量， do_sample=False防止模型瞎想，num_beams=1 
    # !!!!!!!!!!!!!!!!!!!!!!这里直接在模型参数列表塞入参数就不用judge_config了!!!!!!!!!!!!!!!!!!!!!!!!
    # judge_config = {"max_new_tokens": 2, "do_sample": False, "num_beams": 1}

    # !!!!!!!!!!!!!!!!!! 注意 !!!!!!!!!!!!!!!!!!
    # InternLM-XComposer-2.5 模型的输入需要手动在prompt上加图片占位符"<Image_0> <Image_1> ... <Image_n>"（非直接传入视频路径的情况，比较麻烦）
    # 否则模型很可能会只能看到第一帧
    # 生成图片占位符，确保与输入图片数量一致
    # IMG_TOKEN = tokenizer.additional_special_tokens[0]  # "<ImageHere>"
    # image_tags  = " " + " ".join(IMG_TOKEN for _ in frame_paths) 
    # hd = len(frame_paths)  
    
    # image_tags = " " + " ".join(IMG_TOKEN for _ in frame_paths)
    # image_tags = " " + " ".join("<Image>" for _ in range(len(frame_paths)))
    # ids = tokenizer(f"\n{image_tags}").input_ids
    # print("img_token 个数 =", ids.count(tokenizer.convert_tokens_to_ids(IMG_TOKEN)),
    #"hd_num =", len(frame_paths))

    # img_id = tokenizer.convert_tokens_to_ids(IMG_TOKEN)
    # print(tokenizer(image_tags).input_ids.count(img_id))
    # print(len(frame_paths))
    # assert tokenizer(image_tags).input_ids.count(img_id) == len(frame_paths)
    
    # 寄，还要考虑强制指定模型看到的帧数，麻烦死
    # n_frames = len(frame_paths)
    
    # -------- 1. 可答性判定 --------
    if need_judge_answerable:
        # 构建用户的输入文字prompt_yesno，视频帧序列
        if not add_text:
            prompt_yesno = (
                "Is the following question answerable by the video? "
                "Only reply with Yes or No.\n"
                + original_q
            )
        else:
            prompt_yesno = (
                f"{add_text}\n"
                "Is the following question answerable by the video and subtitles? "
                "Only reply with Yes or No.\n"
                + original_q
            )
        
        # 前向推理
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_yes, _ = model.chat(tokenizer, prompt_yesno, image=[chunk_path], meta_instruction="Answer only with Yes or No.", use_meta=False, max_new_tokens=2, do_sample=False, num_beams=1)

        # IMG = tokenizer.additional_special_tokens[0]
        # full = "dummy meta\n" + prompt_yesno          # 用你真实的 meta_instruction & prompt
        # img_id = tokenizer.convert_tokens_to_ids(IMG)
        # print("占位符 =", tokenizer(full).input_ids.count(img_id),
        #     "图片 =", len(frame_paths))
        
        if debug_print:
            print("--- 模型判断是否当前是否具有可回答性：模型的原始输出内容 ---\n", raw_yes)

        if 'yes' in raw_yes.lower():
            answerable = "Yes"
        elif 'no' in raw_yes.lower():
            answerable = "No"
        else:
            print("模型判断是否具有可回答性时，输出内容不符合预期：", raw_yes)
            # print("产生错误的视频编号：", video_data["video_id"])
            answerable = "Yes"
            # raise ValueError(
            #     "模型判断是否具有可回答性时，输出内容不符合预期："
            #     f"原始输出内容为：{raw_yes}"
            # )
            
    else:
        # --------- 如果不需要可答性判定，那就直接设置 answerable 为 "Yes" ---------
        answerable = "Yes"

    
    # -------- 2. 正式问答 --------
    if answerable == "Yes":

        if prev_output:
            prompt_vqa = (
                f"{prev_output}\n{add_text}\n{original_q}"
                if add_text else
                f"{prev_output}\n{original_q}"
            )
        else:
            prompt_vqa = (
                f"{add_text}\n{original_q}" if add_text else
                f"{original_q}"
            )
            
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_vqa, _ = model.chat(tokenizer, prompt_vqa, image=[chunk_path], meta_instruction="Answer the question.", use_meta=False, max_new_tokens=256, do_sample=True, num_beams=1)   
        
        if debug_print:
            print("--- 模型正式回答：模型的原始输出内容 ---\n", raw_vqa)
    else:
        raw_vqa = ""       
        
    # from transformers import AutoTokenizer

    # tok    = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2d5-7b",
    #                                     trust_remote_code=True, use_fast=False)
    # IMG    = tok.additional_special_tokens[0]  # "<ImageHere>"
    # meta   = "Answer only with Yes or No."    # 你传给 chat() 的 meta_instruction
    # query  = image_tags + "\nIs the following question answerable?\n" + original_q
    # full   = meta + "\n" + query

    # count = tok(full).input_ids.count(tok.convert_tokens_to_ids(IMG))
    # print("系统层实际占位符 =", count, 
    #     "图片列表长度 =", len(frame_paths))

    # for p in frame_paths:
    #     os.remove(p)
    
    os.remove(chunk_path)  # 删除临时生成的 MP4 文件
    
    return {"answerable": answerable, "response": raw_vqa}


    # def clean_response(raw_text: str) -> str:
    #     """
    #     只保留 raw_text 中“assistant\n”之后的内容（去掉 system/user/assistant 标签及前面的部分）。
    #     """
    #     lower = raw_text.lower()
    #     tag = "assistant\n"
    #     idx = lower.rfind(tag)
    #     if idx >= 0:
    #         return raw_text[idx + len(tag):].strip()
    #     else:
    #         return raw_text.strip()

    ############ 注意！！！！！！！！！！！！！！！！！！#############
    # 官方的指南建议给模型传入图片的路径，让模型内置的image processor处理，而不是自己处理后的图片列表
    
    # 下面的代码直接将图片转换为张量列表，但存在一些莫名的错误，一些图片在np_frames中是None，但似乎不是处理脚本的问题？
    
    # # pil_frames = [Image.fromarray(x.transpose(1, 2, 0)) for x in np_frames]
    
    # # --------- 将抽到的帧转换为张量（模型只接受完整的路径（这个显然不行，显存会炸） 或张量列表, 单个元素为四维度张亮） ---------
    # # 添加batch维度，防止模型读取错误  expect: [B C H W]
    # tensor_pil_frames = []

    # for f in np_frames:
    #     # 1) uint8 to float32，通道前置 (C, H, W)
    #     t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0   # C,H,W

    #     # 2) 让高和宽都变成 560 的整数倍（向下取整 + 中心裁剪）
    #     C, H, W = t.shape
    #     new_H = 560 * (H // 560)          # 下取整到最近的 560 倍数
    #     new_W = 560 * (W // 560)
    #     h_off = (H - new_H) // 2          # 上下左右各裁掉一半多余像素
    #     w_off = (W - new_W) // 2
    #     t = t[:, h_off : h_off + new_H, w_off : w_off + new_W]  # C,new_H,new_W

    #     # 3) 如想统一到 560×560，可再插一句：
    #     # t = F.interpolate(t.unsqueeze(0), size=(560, 560),
    #     #                   mode="bilinear", align_corners=False)[0]

    #     # 4) 加 batch 维、转 bfloat16、搬到 GPU
    #     t = t.unsqueeze(0)                                # 1,C,H',W'
    #     t = t.to(dtype=torch.bfloat16, device=model.device)

    #     tensor_pil_frames.append(t)
    
    
    
    # # 这里直接将抽帧图片转成临时jpg文件，只将路径传给模型
    # # 仅需把抽好的关键帧保存成临时 jpg，收集路径
    # frame_paths = []
    # for i, t in enumerate(np_frames):
    #     path = f"/tmp/frame_{i:04d}.jpg"
        
    #     arr = (t[0]                                 # → 3×H×W
    #         .permute(1, 2, 0)                    # → H×W×3
    #         .mul(255).clamp(0, 255)
    #         .to(torch.uint8)
    #         .cpu().numpy())
        
    #     Image.fromarray(arr, mode="RGB").save(path, quality=95)
    #     frame_paths.append(path)

    # for i, t in enumerate(tensor_frames):          # t: 1×3×H×W, bfloat16
    #     arr = (t[0].permute(1, 2, 0)              # → H×W×3
    #             .mul(255).clamp(0, 255)         # 到 0–255
    #             .to(torch.uint8)                # uint8
    #             .cpu().numpy())
    #     Image.fromarray(arr, mode="RGB").save(f"{SAVE_DIR}/frame_{i:04d}.jpg", quality=95)


    # raw_yes, _ = model.chat(
    #     tokenizer,
    #     prompt_yesno,
    #     image=frame_paths,      # 直接给路径 list
    #     **gen_cfg
    # )






























    #     chatml_yesno = (
    #         "<|im_start|>system\nYou are a helpful assistant. Reply with Yes or No only.\n<|im_end|>\n"
    #         "<|im_start|>user\n<video>\n" + prompt_yesno + "\n<|im_end|>\n"
    #         "<|im_start|>assistant\n"
    #     )

    #     # 1.1 视觉预处理：messages → (vision_inputs, video_inputs)
    #     messages_yesno = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "video": video_data["video_file"]},
    #             {"type": "text",  "text": prompt_yesno}
    #         ]
    #     }]
    #     vision_inputs_yesno, video_inputs_yesno = process_vision_info(messages_yesno, processor)

    #     if vision_inputs_yesno is not None:
    #         vision_inputs_yesno = vision_inputs_yesno.to(device, dtype=torch.bfloat16)
    #     if video_inputs_yesno is not None:
    #         video_inputs_yesno = video_inputs_yesno.to(device, dtype=torch.bfloat16)

    #     inputs_yesno = processor(
    #         text=[chatml_yesno],
    #         vision_inputs=vision_inputs_yesno,
    #         videos=video_inputs_yesno,
    #         fps=video_data["frame_fps"],
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(device)

    #     with torch.inference_mode():
    #         gen_yesno = model.generate(
    #             **inputs_yesno,
    #             max_new_tokens=gen_cfg["max_new_tokens"],
    #             do_sample     =gen_cfg["do_sample"]
    #         )
    #     raw_yesno = tokenizer.batch_decode(gen_yesno, skip_special_tokens=True)[0]

    #     if debug_print:
    #         print("--- 模型判断是否当前是否具有可回答性：模型的原始输出内容 ---\n", raw_yesno)

    #     # 1.2 取最后一行最后单词
    #     last_line  = raw_yesno.splitlines()[-1].strip() if raw_yesno.splitlines() else raw_yesno.strip()
    #     last_token = last_line.split()[-1].lower() if last_line.split() else raw_yesno.strip().lower()
    #     answerable = "Yes" if last_token == "yes" else ("No" if last_token == "no" else raw_yesno.strip())

    #     if debug_print:
    #         print("--- 模型判断是否当前是否具有可回答性：模型处理后的输出内容 ---\n", answerable)

    #     # --------- 显存优化1 释放掉第一轮可回答性推理的中间数据张量   ---------
    #     del inputs_yesno, gen_yesno, raw_yesno
    #     del vision_inputs_yesno, video_inputs_yesno
    #     torch.cuda.empty_cache()
    #     # ------------------------------------------------------------------
        
        
        
    # else:
    #     answerable = "Yes"

    # # -------- 2. 正式回答 --------
    # if "yes" in answerable.lower():
    #     if prev_output:
    #         prompt_final = f"{prev_output}\n{add_text}\n{original_q}" if add_text else f"{prev_output}\n{original_q}"
    #     else:
    #         prompt_final = f"{add_text}\n{original_q}" if add_text else original_q

    #     chatml_final = (
    #         "<|im_start|>system\nYou are a helpful assistant. Provide a concise answer based on the video.\n<|im_end|>\n"
    #         "<|im_start|>user\n<video>\n" + prompt_final + "\n<|im_end|>\n"
    #         "<|im_start|>assistant\n"
    #     )

    #     # 2.1 视觉预处理
    #     messages_final = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "video": video_data["video_file"]},
    #             {"type": "text",  "text": prompt_final}
    #         ]
    #     }]
    #     vision_inputs_final, video_inputs_final = process_vision_info(messages_final, processor)
    #     if vision_inputs_final is not None:
    #         vision_inputs_final = vision_inputs_final.to(device, dtype=torch.bfloat16)
    #     if video_inputs_final is not None:
    #         video_inputs_final  = video_inputs_final.to(device, dtype=torch.bfloat16)

    #     inputs_final = processor(
    #         text=[chatml_final],
    #         vision_inputs=vision_inputs_final,
    #         videos=video_inputs_final,
    #         fps=video_data["frame_fps"],
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(device)

    #     with torch.inference_mode():
    #         gen_final = model.generate(
    #             **inputs_final,
    #             max_new_tokens=gen_cfg["max_new_tokens"],
    #             do_sample     =gen_cfg["do_sample"]
    #         )
    #     raw_final = tokenizer.batch_decode(gen_final, skip_special_tokens=True)[0]

    #     if debug_print:
    #         print("--- 模型正式回答：模型的原始输出内容 ---\n", raw_final)

    #     response = clean_response(raw_final)

    #     if debug_print:
    #         print("--- 模型正式：模型处理后的输出内容 ---\n", response)

    #     # --------- 显存优化2 释放掉第一轮可回答性推理的中间数据张量   ---------
    #     del inputs_final, gen_final, raw_final
    #     del vision_inputs_final, video_inputs_final
    #     torch.cuda.empty_cache()
    #     # ------------------------------------------------------------------
        
    # else:
    #     response = None

    # return {"answerable": answerable, "response": response}

      