import os
import traceback

import torch

import x_vars_demo_with_streamlit as app


def run():
    results = {}
    video_path = os.path.join("dataset", "Test", "action_0", "clip_0.mp4")
    question = "这个动作是否犯规？请给出简短判罚建议。"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ENV] python={os.sys.version}")
    print(f"[ENV] torch={torch.__version__}")
    print(f"[ENV] device={device}")
    print(f"[ENV] video_exists={os.path.exists(video_path)} path={video_path}")

    # 1) 模型加载
    try:
        model, vision_tower_hf, clip_net, tokenizer, image_processor = app.load_all_models(
            app.BASE_MODEL_DIR,
            app.LOCAL_TOKENIZER_DIR if os.path.isdir(app.LOCAL_TOKENIZER_DIR) else None,
            app.LOCAL_CLIP_DIR if os.path.isdir(app.LOCAL_CLIP_DIR) else None,
            app.CLIP_CLASSIFIER_WEIGHTS if os.path.isfile(app.CLIP_CLASSIFIER_WEIGHTS) else None,
        )
        results["model_load"] = "ok"
        print("[OK] model_load")
    except Exception as e:
        results["model_load"] = f"fail: {e}"
        print(f"[FAIL] model_load: {e}")
        traceback.print_exc()
        return results

    # 2) 视频特征提取
    try:
        frames = app.extract_frames_to_tensor(video_path, image_processor=None, max_frames=8, device=device)
        feats = app.process_clip_features(frames, device, clip_bs=1, video_token_len=128)
        assert feats is not None and len(feats.shape) == 3 and feats.shape[0] == 1
        results["feature_extract"] = f"ok shape={tuple(feats.shape)}"
        print(f"[OK] feature_extract shape={tuple(feats.shape)}")
    except Exception as e:
        results["feature_extract"] = f"fail: {e}"
        print(f"[FAIL] feature_extract: {e}")
        traceback.print_exc()
        return results

    # 3) 模型推理
    try:
        prompt = app.build_prompt(question, model, video_token_len=128)
        answer = app.model_generate_simple(
            model=model,
            tokenizer=tokenizer,
            video_spatio_temporal_features=feats,
            prompt=prompt,
            temperature=0.2,
            max_new_tokens=48,
            do_sample=False,
            debug=False,
        )
        results["inference"] = f"ok len={len(answer)}"
        print(f"[OK] inference len={len(answer)}")
        print(f"[OUT] {answer[:300]}")
    except Exception as e:
        results["inference"] = f"fail: {e}"
        print(f"[FAIL] inference: {e}")
        traceback.print_exc()

    # 4) RAG 初始化+检索
    try:
        rag = app.FootballRAGSystem(device="cpu")
        init_ok = rag.initialize()
        search_results = rag.search("黄牌 红牌 犯规", top_k=3) if init_ok else []
        results["rag"] = f"ok init={init_ok} hits={len(search_results)}"
        print(f"[OK] rag init={init_ok} hits={len(search_results)}")
        if search_results:
            top = search_results[0]
            print(f"[RAG_TOP] score={top.get('score')} text={str(top.get('text', ''))[:120]}")
    except Exception as e:
        results["rag"] = f"fail: {e}"
        print(f"[FAIL] rag: {e}")
        traceback.print_exc()

    return results


if __name__ == "__main__":
    final = run()
    print("\n=== SUMMARY ===")
    for k, v in final.items():
        print(f"{k}: {v}")
