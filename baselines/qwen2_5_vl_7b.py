import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info  # 

from utils.video_utils import load_video  # 你之前提供的 load_video 实现

def clean_response(raw_text: str):
    """
    从模型输出的 raw_text 中剥离所有角色标签，只保留真正的“assistant”回答部分。

    示例模型的原始输出内容：
        system\n
        You are a helpful assistant.\n
        user\n
        {What event is taking place involving NASA?}\n
        assistant\n
        {"The event involves NASA's Administrator, Bolden, visiting ... "}

    处理思路：
    1. 查找字符串中最后一次出现的 “assistant\n” （不区分大小写）。
    2. 如果找到了，就把它后面的所有内容全部返回；否则直接返回 raw_text。
    3. 去掉首尾空白后返回。
    """
    lower = raw_text.lower()
    tag = "assistant\n"
    idx = lower.rfind(tag)
    if idx >= 0:
        # “assistant\n” 出现的位置 + len(tag) 即为真正回答开始的位置
        start = idx + len(tag)
        return raw_text[start:].strip()
    else:
        # 没有找到 “assistant\n”，直接返回原始文本（strip 后）
        return raw_text.strip()

def load_model(llm_pretrained, attn_implementation):
    """
    attn_implementation 暂时忽略（Qwen2.5-VL-7B-Instruct 构造函数不接受 flash 参数）。

    返回：
      {
        "model": <Qwen2_5_VLForConditionalGeneration 实例>,
        "tokenizer": <AutoTokenizer 实例>,
        "processor": <AutoProcessor 实例>,
        "generation_config": {"max_new_tokens": 256, "do_sample": True}
      }
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        llm_pretrained,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"  # 自动分配到可用的 GPU
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        llm_pretrained,
        trust_remote_code=True,
        use_fast=False  # 保证与 processor 匹配 
    )
    processor = AutoProcessor.from_pretrained(
        llm_pretrained,
        trust_remote_code=True
    )
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
    参数：
      - baseline_model: load_model 返回的 dict，包含 "model","tokenizer","processor","generation_config"
      - video_data: {
            "video_file": <str 视频路径>,
            "start_sec": <int chunk 开始秒>,
            "end_sec": <int chunk 结束秒>,
            "frame_resolution": <int 缩放后边长>,
            "frame_fps": <int 抽帧帧率>
        }
      - text_data: {
            "original_question":    <str 原问题>,
            "additional_text_input":<str 额外文本，可空>,
            "previous_turns_output":<str 历史回答，仅 openai_api 用>
        }
      - need_judge_answerable: bool，是否先做“可答性判定”
      - debug_print: bool，是否打印所有中间 prompt/output

    返回（与 InternVL 一致）：
      {
        "answerable": <"Yes" 或 "No" 或模型输出其他关键词>,
        "response":   <最终回答字符串 或 None>
      }
    """
    device = "cuda"
    model = baseline_model["model"]
    tokenizer = baseline_model["tokenizer"]
    processor = baseline_model["processor"]
    generation_config = baseline_model["generation_config"]

    # -------------------------------------------------
    # 1. 准备视频和文本的原始数据 
    # -------------------------------------------------
    
    # 视频数据：抽帧：调用 load_video，得到 List[np.ndarray]；若为空则不做视觉输入 
    pixel_list = load_video(
        video_data["video_file"],
        start_sec=video_data["start_sec"],
        end_sec=video_data["end_sec"],
        output_resolution=video_data["frame_resolution"],
        output_fps=video_data["frame_fps"]
    )
    pil_frames = []
    if len(pixel_list) > 0:
        from PIL import Image
        for frame_np in pixel_list:
            # frame_np.shape = (3, H, W) → 转为 (H, W, 3) → PIL.Image
            pil_frames.append(Image.fromarray(frame_np.transpose(1, 2, 0)))
    # 如果没抽到帧，pil_frames 保持空列表

    # --------- 显存优化 1 释放掉原始的视频序列 ---------
    del pixel_list
    # -----------------------------------------------
    
    original_question = text_data["original_question"]
    additional_text_input = text_data.get("additional_text_input", "")
    previous_turns_output = text_data.get("previous_turns_output", "")

    # -------------------------------------------------
    # 2. “可答性判定” (Yes/No) → 产生判断yes or no的模型输入，即prompt_yesno
    # -------------------------------------------------
    if need_judge_answerable:
        
        # 产生文字信息prompt
        if not additional_text_input.strip():
            prompt_yesno = (
                "Is the following question answerable by the video? Only reply with Yes or No.\n"
                + original_question
            )
        else:
            prompt_yesno = (
                additional_text_input.strip() + "\n"
                + "Is the following question answerable by the video and subtitles? Only reply with Yes or No.\n"
                + original_question
            )
        
        # Qwen模型的Chat Template，要求用户消息包含视频和文本内容
        user_msg_yesno = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_data["video_file"]},
                {"type": "text",  "text": prompt_yesno}
            ]
        }
        messages_yesno = [user_msg_yesno]

        # 2.1 传 messages 给 process_vision_info以预处理输入视觉数据，返回 (image_inputs, video_inputs)
        image_inputs_yesno, video_inputs_yesno = process_vision_info(messages_yesno)
        # 

        # 2.2 用 apply_chat_template 拼出完整的text prompt
        text_yesno = processor.apply_chat_template(
            messages_yesno,
            tokenize=False,
            add_generation_prompt=True
        )
        
        with torch.inference_mode():
            # 2.3 把 text + image_inputs + video_inputs + fps 传给 processor，得到所有输入张量
            inputs_yesno = processor(
                text=[text_yesno],
                images=image_inputs_yesno if image_inputs_yesno else None,
                videos=video_inputs_yesno if video_inputs_yesno else None,
                fps=video_data["frame_fps"],  # 关键：让模型知道实际帧率 
                padding=True,
                return_tensors="pt"
            ).to(device)
            # 

            # 2.4 输入模型执行yes or no推理，生成 Yes/No 回答
            generated_yesno = model.generate(
                **inputs_yesno,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"]
            )
            # 解码原始输出内容，得到自然语言回答
            decoded_yesno = tokenizer.batch_decode(
                generated_yesno, skip_special_tokens=True
            )[0].strip()

        # 后处理 “answerable” 输出：若包含 “yes” → “Yes”；若包含 “no” → “No”；否则报错，说明模型没有理解意思
        lower_yesno = decoded_yesno.lower()
        
        
        
        # —— 新的后处理 ——  
        # 1) 按行拆分，取最后一行  
        lines = decoded_yesno.splitlines()
        last_line = lines[-1].strip() if lines else decoded_yesno.strip()
        
        # 2) 按空白字符拆分最后一行，取最后一个 token  
        tokens = last_line.split()
        if not tokens:
            raise ValueError(f"Cannot parse answerable token from model output: '{decoded_yesno}'")
        last_token = tokens[-1].strip().lower()

        # 3) 判断最后一个 token 是否为 “yes” 或 “no”  
        if last_token == "yes":
            answerable_response = "Yes"
        elif last_token == "no":
            answerable_response = "No"
        else:
            raise ValueError(
                f"Expected final token 'Yes' or 'No', but got '{tokens[-1]}' in model output: '{decoded_yesno}'"
            )
        
        # if "yes" in lower_yesno:
        #     answerable_response = "Yes"
        # elif "no" in lower_yesno:
        #     answerable_response = "No"
        # else:
        #     raise ValueError("Model did not understand the question about answerability.")

        # debug_print 打印中间结果
        if debug_print:
            print("===== 可答性 判定 Prompt 用户询问MLLM =====")
            print(user_msg_yesno["content"][1]["text"])
            print("===== 原始 MLLM给出的模型输出 =====")
            print(decoded_yesno)
            print("===== 处理后 answerable_response =====")
            print(answerable_response)
        
        # --------- 显存优化 2 释放掉第一轮可回答性推理的中间数据张量   ---------
        del inputs_yesno, generated_yesno, decoded_yesno
        if 'image_inputs_yesno' in locals():
            del image_inputs_yesno
        if 'video_inputs_yesno' in locals():
            del video_inputs_yesno
        torch.cuda.empty_cache()  # 建议清理一下碎片
        # --------------------------------------------------------------------
        
        
    else:
        answerable_response = "Yes"

    # -------------------------------------------------
    # 3. 如果包含 “yes”，走正式回答；否则 response=None
    # -------------------------------------------------
    if "yes" in answerable_response.lower():
        
        # 3.1 构造正式回答的 文本text prompt
        if not previous_turns_output.strip():
            if not additional_text_input.strip():
                final_text = original_question
            else:
                final_text = additional_text_input.strip() + "\n" + original_question
        else:
            if not additional_text_input.strip():
                final_text = previous_turns_output.strip() + "\n" + original_question
            else:
                final_text = (
                    previous_turns_output.strip()
                    + "\n"
                    + additional_text_input.strip()
                    + "\n"
                    + original_question
                )
        # Qwen模型的Chat Template，要求用户消息包含视频和文本内容
        user_msg_final = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_data["video_file"]},
                {"type": "text",  "text": final_text}
            ]
        }
        messages_final = [user_msg_final]

        # 3.2传 messages 给 process_vision_info以预处理输入视觉数据 process_vision_info → 提取 image_inputs_final, 
        # video_inputs_final
        image_inputs_final, video_inputs_final = process_vision_info(messages_final)


        # 3.3 apply_chat_template
        text_final = processor.apply_chat_template(
            messages_final,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.inference_mode():
            # 3.4 processor(...) 合并所有数据，并传入 GPU
            inputs_final = processor(
                text=[text_final],
                images=image_inputs_final if image_inputs_final else None,
                videos=video_inputs_final if video_inputs_final else None,
                fps=video_data["frame_fps"],
                padding=True,
                return_tensors="pt"
            ).to(device)
            # 

            # 3.5 调用 model.generate 生成最终回答
            generated_final = model.generate(
                **inputs_final,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"]
            )
            decoded_final = tokenizer.batch_decode(
                generated_final, skip_special_tokens=True
            )[0]    # 这里先不要 .strip()，让 clean_response 去掉首尾空白
        
        # 3.6 clean_response 处理模型输出，掉一些没用的信息，只保留模型的回答内容
        response = clean_response(decoded_final)

        # debug_print 打印中间结果
        if debug_print:
            print("===== 正式问答 Prompt 用户询问MLLM =====")
            print(user_msg_final["content"][1]["text"])
            print("===== 原始 MLLM给出的模型输出 =====")
            print(decoded_final)
            print("===== 处理后 model_response =====")
            print(response)
        
        # --------- 显存优化 3 释放掉正式回答推理的中间数据张量   ---------
        del inputs_final, generated_final, raw_final
        if 'image_inputs_final' in locals():
            del image_inputs_final
        if 'video_inputs_final' in locals():
            del video_inputs_final
        torch.cuda.empty_cache()
        # --------------------------------------------------------------
        
    else:
        response = None

    return {
        "answerable": answerable_response,
        "response": response
    }
