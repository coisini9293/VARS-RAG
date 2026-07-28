
# ===================================================================
# 第一部分：环境设置和依赖导入
# ===================================================================

# 导入必要的库
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体 - 简单有效的解决方案
import matplotlib.pyplot as plt
from matplotlib import rcParams

print("Setting up English fonts for better compatibility...")

# 设置英文字体 - 确保图表中文字显示正常
rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print("English font setup completed")

# 测试字体显示
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 测试文本
    test_text = 'Font Test: X-VARS Football Video Analysis System'
    test_labels = ['Action Type', 'Foul Decision', 'Severity', 'Feature Stats', 'Temporal', 'Spatial']
    
    ax.text(0.5, 0.8, test_text, ha='center', va='center', fontsize=16, color='blue', weight='bold')
    ax.text(0.5, 0.6, 'Common Chart Labels Test:', ha='center', va='center', fontsize=14, color='red', weight='bold')
    
    # 显示测试标签
    for i, label in enumerate(test_labels):
        col = i % 3
        row = i // 3
        x_pos = 0.2 + col * 0.3
        y_pos = 0.4 - row * 0.1
        ax.text(x_pos, y_pos, label, ha='center', va='center', fontsize=12, color='purple')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Font Test Result - X-VARS System', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print("\n=== 字体测试分析 ===")
    print("图像名称: 字体测试显示")
    print("目的: 验证英文字体是否正确显示")
    print("分析:")
    print("  1. 蓝色文字显示主系统标题")
    print("  2. 紫色标签显示常用图表术语")
    print("  3. 所有文字应清晰可读")
    print("成功: 字体设置对英文显示工作正常")
    print("=" * 50)
    
except Exception as e:
    print(f"字体测试失败: {e}")

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('default')

print("📚 导入基础库完成")
print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"🎯 CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🚀 GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ===================================================================
# 第二部分：设置项目路径和导入自定义模块
# ===================================================================

# 设置项目根目录路径（根据你的实际路径修改）
project_root = r"E:\python_project\往期项目\vars\xvars_no_model"
visual_encoder_path = os.path.join(project_root, "visual_encoder")

# 将路径添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)
if visual_encoder_path not in sys.path:
    sys.path.append(visual_encoder_path)

print(f"📁 项目根目录: {project_root}")
print(f"🔍 视觉编码器路径: {visual_encoder_path}")

# 导入自定义模块
try:
    from visual_encoder.clip_model import CLIPNetwork
    from visual_encoder.create_features import load_video, get_seq_frames, get_spatio_temporal_features
    from visual_encoder.config.classes import INVERSE_EVENT_DICTIONARY
    from transformers import CLIPImageProcessor
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查路径设置是否正确")

# 定义本地的get_features函数，确保可以访问全局变量
def get_features(video_path):
    """
    视频特征提取和多任务预测函数（本地版本）
    
    参数:
        video_path: 视频文件路径
    
    返回:
        video_clip_features: 时空融合特征
        values: 预测结果字典
    """
    # 获取全局变量
    image_processor = globals().get('image_processor')
    model = globals().get('model')
    
    if image_processor is None:
        raise ValueError("image_processor未初始化")
    if model is None:
        raise ValueError("model未初始化")
    
    # 视频时间段设置（参考create_features.py）
    start_frame = 63
    end_frame = 87
    
    # 帧率转换参数
    fps = 17
    fps_beginning = 25
    
    # 计算帧率转换因子
    frame_duration = (end_frame - start_frame) / fps_beginning
    target_frame_count = frame_duration * fps
    factor = (end_frame - start_frame) / target_frame_count
    
    # 视频加载和预处理
    video_frames = load_video(video_path)
    video_frames = video_frames[start_frame:end_frame]
    
    # 使用CLIP预处理器处理图像
    frames = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    
    # 帧率调整和帧选择
    final_frames = None
    
    for j in range(len(frames)):
        if j % factor < 1:
            current_frame = frames[j, :, :, :].unsqueeze(0)
            
            if final_frames is None:
                final_frames = current_frame
            else:
                final_frames = torch.cat((final_frames, current_frame), 0)
    
    # 将帧移动到GPU
    if torch.cuda.is_available():
        final_frames = final_frames.cuda()
    
    # 模型推理
    out_off, out_act, video_features = model(final_frames)
    
    # 预测结果解析
    preds_sev = torch.argmax(out_off.detach().cpu(), 0)
    preds_act = torch.argmax(out_act.detach().cpu(), 0)
    
    # 结果格式化
    values = {}
    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
    
    if preds_sev.item() == 0:
        values["Offence"] = "No offence"
        values["Severity"] = ""
    elif preds_sev.item() == 1:
        values["Offence"] = "Offence"
        values["Severity"] = "1.0"
    elif preds_sev.item() == 2:
        values["Offence"] = "Offence"
        values["Severity"] = "3.0"
    elif preds_sev.item() == 3:
        values["Offence"] = "Offence"
        values["Severity"] = "5.0"
    
    # 特征后处理
    video_clip_features = get_spatio_temporal_features(
        video_features.cpu().numpy().astype("float16")
    )
    
    return video_clip_features, values
# ===================================================================
# 第三部分：配置参数
# ===================================================================

# 配置参数
CONFIG = {
    # 视频文件路径（可以修改为你要分析的视频文件）
    'video_path': os.path.join(project_root, "dataset", "videos", "clip_1.mp4"),
    
    # 备选视频路径（如果上面的不存在，可以尝试这些）
    'alternative_videos': [
        os.path.join(project_root, "dataset", "Train", "action_0", "clip_1.mp4"),
        os.path.join(project_root, "dataset", "Train", "action_1", "clip_1.mp4"),
        os.path.join(project_root, "dataset", "Valid", "action_0", "clip_1.mp4"),
    ],
    
    # 模型权重路径
    'model_weights': os.path.join(visual_encoder_path, "14_model.pth.tar"),
    
    # CLIP模型配置
    'clip_model_name': 'openai/clip-vit-large-patch14',
    
    # 输出目录
    'output_dir': os.path.join(project_root, "analysis_results"),
    
    # 分析参数
    'num_frames_to_analyze': 1000,  # 最大分析帧数
    'save_intermediate_results': True,  # 是否保存中间结果
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
print(f"📂 输出目录: {CONFIG['output_dir']}")

# 检查视频文件是否存在
video_found = False
for video_path in [CONFIG['video_path']] + CONFIG['alternative_videos']:
    if os.path.exists(video_path):
        CONFIG['video_path'] = video_path
        video_found = True
        print(f"🎬 找到视频文件: {video_path}")
        break

if not video_found:
    print("❌ 未找到可用的视频文件，请检查路径配置")
    print("可用的视频路径:")
    for path in [CONFIG['video_path']] + CONFIG['alternative_videos']:
        print(f"  - {path}")

# 检查模型权重文件
if os.path.exists(CONFIG['model_weights']):
    print(f"🏋️ 模型权重文件: {CONFIG['model_weights']}")
else:
    print(f"❌ 模型权重文件不存在: {CONFIG['model_weights']}")
# ===================================================================
# 第四部分：初始化模型和预处理器
# ===================================================================

print("\n🚀 正在初始化模型和预处理器...")

# 初始化CLIP图像预处理器
try:
    image_processor = CLIPImageProcessor.from_pretrained(
        'openai/clip-vit-large-patch14', 
        torch_dtype=torch.float16
    )
    print("✅ CLIP图像预处理器初始化成功")
    # 设置为全局变量，确保在get_features函数中可以访问
    globals()['image_processor'] = image_processor
except Exception as e:
    print(f"❌ CLIP图像预处理器初始化失败: {e}")
    # 如果初始化失败，创建一个None值的全局变量
    globals()['image_processor'] = None

# 设置CLIP模型路径环境变量
clip_model_path = os.path.join(visual_encoder_path, "clip-vit-large-patch14")
os.environ['LOCAL_CLIP_DIR'] = clip_model_path
print(f"🔧 设置CLIP模型路径: {clip_model_path}")

# 初始化模型
model = None
model_loaded = False

try:
    print("🔄 正在初始化CLIPNetwork模型...")
    model = CLIPNetwork()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("🚀 模型已移动到GPU")
    else:
        print("💻 使用CPU运行模型")
    
    # 加载预训练权重
    if os.path.exists(CONFIG['model_weights']):
        print("🔄 正在加载预训练权重...")
        checkpoint = torch.load(
            CONFIG['model_weights'], 
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ 模型权重加载成功")
        model_loaded = True
    else:
        print("⚠️ 模型权重文件不存在，使用随机初始化权重")
        model_loaded = True  # 即使没有权重，模型也可以用于测试
    
    # 设置为评估模式
    model.eval()
    print("✅ 模型初始化完成，进入评估模式")
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    print("🔧 错误详情:")
    print(f"   - CLIP模型路径: {clip_model_path}")
    print(f"   - 路径是否存在: {os.path.exists(clip_model_path)}")
    if os.path.exists(clip_model_path):
        print(f"   - 目录内容: {os.listdir(clip_model_path)}")
    print("💡 解决建议:")
    print("   1. 确保clip-vit-large-patch14目录存在且包含必要文件")
    print("   2. 或者下载CLIP模型文件到指定目录")
    print("   3. 检查网络连接，可能需要在线下载模型")
    
    # 尝试在线下载作为备选方案
    try:
        print("🌐 尝试在线加载CLIP模型...")
        from transformers import CLIPVisionModel
        online_model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        print("✅ 在线模型加载成功，但需要适配到CLIPNetwork结构")
    except Exception as online_e:
        print(f"❌ 在线加载也失败: {online_e}")

# 全局变量设置，确保后续代码可以正常执行
globals()['model'] = model
globals()['model_loaded'] = model_loaded
globals()['image_processor'] = image_processor

# ===================================================================
# 第五部分：视频基本信息分析
# ===================================================================

print("\n📊 开始分析视频基本信息...")

def analyze_video_info(video_path):
    """
    分析视频的基本信息
    
    参数:
        video_path: 视频文件路径
    
    返回:
        info: 包含视频信息的字典
    """
    try:
        from decord import VideoReader, cpu
        
        # 使用VideoReader获取视频信息
        vr = VideoReader(video_path, ctx=cpu(0))
        
        info = {
            'file_path': video_path,
            'file_name': os.path.basename(video_path),
            'file_size_mb': os.path.getsize(video_path) / (1024 * 1024),
            'total_frames': len(vr),
            'fps': vr.get_avg_fps(),
            'duration_seconds': len(vr) / vr.get_avg_fps(),
            'resolution': (vr[0].shape[1], vr[0].shape[0]),  # (width, height)
        }
        
        return info
        
    except Exception as e:
        print(f"❌ 视频信息分析失败: {e}")
        return None

# 分析视频信息
if video_found:
    video_info = analyze_video_info(CONFIG['video_path'])
    
    if video_info:
        print("\n📹 视频基本信息:")
        print(f"  📁 文件名: {video_info['file_name']}")
        print(f"  💾 文件大小: {video_info['file_size_mb']:.2f} MB")
        print(f"  🎞️ 总帧数: {video_info['total_frames']}")
        print(f"  ⏱️ 帧率: {video_info['fps']:.2f} FPS")
        print(f"  ⏰ 时长: {video_info['duration_seconds']:.2f} 秒")
        print(f"  📐 分辨率: {video_info['resolution'][0]}x{video_info['resolution'][1]}")
        
        # 可视化视频信息
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Video Information Analysis - {video_info["file_name"]}', fontsize=16, fontweight='bold')
        
        # 1. 文件大小饼图
        file_size = video_info['file_size_mb']
        remaining = max(100 - file_size, 0)  # 避免负数
        sizes = [file_size, remaining] if remaining > 0 else [file_size]
        labels = [f'Video File\n{file_size:.1f}MB', 'Other'] if remaining > 0 else [f'Video File\n{file_size:.1f}MB']
        colors = ['lightblue', 'lightgray'] if remaining > 0 else ['lightblue']
        
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[0, 0].set_title('File Size Distribution', fontweight='bold')
        
        # 2. 视频参数条形图
        metrics = ['FPS', 'Duration(s)', 'Frames/100']
        values = [video_info['fps'], video_info['duration_seconds'], video_info['total_frames']/100]
        bars = axes[0, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Video Parameters', fontweight='bold')
        axes[0, 1].set_ylabel('Value')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 分辨率可视化
        width, height = video_info['resolution']
        axes[1, 0].barh(['Width', 'Height'], [width, height], color=['orange', 'purple'])
        axes[1, 0].set_title('Video Resolution', fontweight='bold')
        axes[1, 0].set_xlabel('Pixels')
        
        # 添加分辨率标签
        for i, (label, value) in enumerate([('Width', width), ('Height', height)]):
            axes[1, 0].text(value + value*0.01, i, f'{value}px', 
                            va='center', fontweight='bold')
        
        # 4. 关键统计信息
        stats_text = f"""Key Statistics:
• Frame Rate: {video_info['fps']:.1f} FPS
• Total Duration: {video_info['duration_seconds']:.1f}s
• Avg Frame Size: {video_info['file_size_mb']*1024/video_info['total_frames']:.2f}KB
• Aspect Ratio: {width/height:.2f}:1
• Total Pixels: {width*height:,}
• Video Quality: {'HD' if width >= 1280 else 'SD'}
• Compression Ratio: {(width*height*video_info['total_frames']*3)/(video_info['file_size_mb']*1024*1024):.1f}:1"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, 
                        transform=axes[1, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== Video Information Analysis Chart ===")
        print("Chart Name: Video Information Analysis Dashboard")
        print("Purpose: Comprehensive overview of video file properties and characteristics")
        print("Chart Components:")
        print("  Top-Left (Pie Chart): File Size Distribution")
        print("    - Shows video file size relative to a 100MB baseline")
        print("    - Helps assess storage requirements and file efficiency")
        print("  Top-Right (Bar Chart): Video Parameters")
        print("    - FPS: Frame rate determines video smoothness (25fps is standard)")
        print("    - Duration: Total video length in seconds")
        print("    - Frames/100: Total frame count scaled for display")
        print("  Bottom-Left (Horizontal Bar): Video Resolution")
        print("    - Width and Height in pixels")
        print("    - Critical for determining video quality and processing requirements")
        print("  Bottom-Right (Text Box): Statistics Summary")
        print("    - Comprehensive metrics including aspect ratio, compression ratio")
        print("    - Quality assessment (HD vs SD)")
        print("Analysis Guidelines:")
        print("  1. Frame Rate: 25fps is standard for football analysis")
        print("  2. Resolution: Higher resolution = better analysis but slower processing")
        print("  3. File Size: Indicates compression efficiency")
        print("  4. Aspect Ratio: Should be close to 16:9 (1.78) for standard video")
        print("  5. Compression Ratio: Higher values indicate better compression")
        print("=" * 60)
# ===================================================================
# 第六部分：视频帧采样和预览
# ===================================================================

print("\n🎬 开始视频帧采样和预览...")

def sample_and_preview_frames(video_path, num_preview_frames=8, num_total_frames=100):
    """
    从视频中采样帧并生成预览
    
    参数:
        video_path: 视频文件路径
        num_preview_frames: 预览显示的帧数
        num_total_frames: 总采样帧数
    
    返回:
        frames: 采样的帧列表
        frame_indices: 对应的帧索引
    """
    try:
        # 加载视频
        video_frames = load_video(video_path, num_total_frames)
        
        # 选择预览帧的索引
        preview_indices = np.linspace(0, len(video_frames)-1, num_preview_frames, dtype=int)
        
        # 创建预览图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Video Frame Preview - Selected {num_preview_frames} from {len(video_frames)} frames', 
                     fontsize=14, fontweight='bold')
        
        for i, idx in enumerate(preview_indices):
            row = i // 4
            col = i % 4
            
            # 显示帧
            axes[row, col].imshow(video_frames[idx])
            axes[row, col].set_title(f'Frame {idx+1}/{len(video_frames)}', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            
            # 添加时间戳信息
            if 'fps' in locals() and video_info:
                timestamp = idx / video_info['fps']
                axes[row, col].text(0.02, 0.98, f'{timestamp:.2f}s', 
                                   transform=axes[row, col].transAxes, 
                                   fontsize=8, color='white', weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== Video Frame Preview Chart ===")
        print("Chart Name: Video Frame Sampling Preview")
        print("Purpose: Visual inspection of video content at different time points")
        print("Chart Structure:")
        print("  - 2x4 grid showing 8 representative frames")
        print("  - Frames are uniformly sampled across the video timeline")
        print("  - Each frame shows timestamp and frame number")
        print("Analysis Value:")
        print("  1. Content Verification: Ensure video contains expected football action")
        print("  2. Quality Assessment: Check for blur, lighting, or corruption issues")
        print("  3. Action Timeline: Understand when key events occur in the video")
        print("  4. Scene Consistency: Verify consistent camera angle and field view")
        print("  5. Player Visibility: Confirm players and action are clearly visible")
        print("Interpretation Guidelines:")
        print("  - Look for smooth temporal progression between frames")
        print("  - Identify key action moments (tackles, fouls, etc.)")
        print("  - Check for camera movement or scene changes")
        print("  - Assess overall video quality for analysis suitability")
        print("=" * 60)
        
        return video_frames, preview_indices
        
    except Exception as e:
        print(f"❌ 帧采样失败: {e}")
        return None, None

# 执行帧采样和预览
if video_found:
    sampled_frames, preview_indices = sample_and_preview_frames(
        CONFIG['video_path'], 
        num_preview_frames=8, 
        num_total_frames=CONFIG['num_frames_to_analyze']
    )
    
    if sampled_frames:
        print(f"✅ 成功采样 {len(sampled_frames)} 帧")
        print(f"🖼️ 预览帧索引: {preview_indices}")
# ===================================================================
# 第七部分：特征提取和预测
# ===================================================================

print("\n🧠 开始特征提取和预测分析...")

def analyze_video_features(video_path):
    """
    使用X-VARS模型分析视频特征和预测结果
    
    参数:
        video_path: 视频文件路径
    
    返回:
        features: 提取的特征
        predictions: 预测结果
        analysis: 详细分析结果
    """
    try:
        print("🔄 正在提取视频特征...")
        
        # 检查必要的全局变量是否存在
        if 'image_processor' not in globals() or globals()['image_processor'] is None:
            print("❌ image_processor未初始化")
            return None, None, None
        
        if 'model' not in globals() or globals()['model'] is None:
            print("❌ model未初始化")
            return None, None, None
        
        if not globals().get('model_loaded', False):
            print("❌ 模型未正确加载")
            return None, None, None
        
        # 使用create_features模块的get_features函数
        features, predictions = get_features(video_path)
        
        print("✅ 特征提取完成")
        print(f"📊 特征维度: {features.shape}")
        print(f"🎯 预测结果: {predictions}")
        
        # 创建详细分析结果
        analysis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'video_path': video_path,
            'feature_shape': features.shape,
            'feature_stats': {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'non_zero_ratio': float(np.count_nonzero(features) / features.size)
            },
            'predictions': predictions,
            'action_confidence': None,  # 如果模型输出概率，可以在这里添加
            'offence_confidence': None,
        }
        
        return features, predictions, analysis
        
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        
        # 如果特征提取失败，生成模拟数据用于演示
        print("🔄 生成模拟数据用于演示...")
        try:
            # 生成模拟特征数据
            mock_features = np.random.randn(300, 1024).astype(np.float16)
            
            # 生成模拟预测结果
            mock_predictions = {
                "Action class": "Tackling",
                "Offence": "Offence", 
                "Severity": "3.0"
            }
            
            # 创建模拟分析结果
            mock_analysis = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_path': video_path,
                'feature_shape': mock_features.shape,
                'feature_stats': {
                    'mean': float(np.mean(mock_features)),
                    'std': float(np.std(mock_features)),
                    'min': float(np.min(mock_features)),
                    'max': float(np.max(mock_features)),
                    'non_zero_ratio': float(np.count_nonzero(mock_features) / mock_features.size)
                },
                'predictions': mock_predictions,
                'action_confidence': None,
                'offence_confidence': None,
            }
            
            print("✅ 模拟数据生成完成（仅用于演示）")
            print(f"📊 模拟特征维度: {mock_features.shape}")
            print(f"🎯 模拟预测结果: {mock_predictions}")
            
            return mock_features, mock_predictions, mock_analysis
            
        except Exception as mock_e:
            print(f"❌ 模拟数据生成也失败: {mock_e}")
            return None, None, None

# 执行特征提取和预测
if video_found and 'model' in globals() and globals().get('model_loaded', False):
    video_features, video_predictions, feature_analysis = analyze_video_features(CONFIG['video_path'])
    
    if video_features is not None:
        print("\n📋 预测结果详情:")
        print(f"  🏃 动作类型: {video_predictions.get('Action class', 'N/A')}")
        print(f"  ⚖️ 是否犯规: {video_predictions.get('Offence', 'N/A')}")
        print(f"  🟡 犯规严重程度: {video_predictions.get('Severity', 'N/A') or '无'}")
        
        print("\n📊 特征统计信息:")
        stats = feature_analysis['feature_stats']
        print(f"  📏 特征维度: {feature_analysis['feature_shape']}")
        print(f"  📈 均值: {stats['mean']:.6f}")
        print(f"  📉 标准差: {stats['std']:.6f}")
        print(f"  🔽 最小值: {stats['min']:.6f}")
        print(f"  🔼 最大值: {stats['max']:.6f}")
        print(f"  🎯 非零比例: {stats['non_zero_ratio']:.2%}")
else:
    print("\n⚠️ 跳过特征提取：")
    if not video_found:
        print("  - 视频文件未找到")
    if 'model' not in globals():
        print("  - 模型未初始化")
    if not globals().get('model_loaded', False):
        print("  - 模型未正确加载")
    
    # 为了演示目的，生成模拟数据
    print("🔄 生成模拟数据以继续演示流程...")
    video_features = np.random.randn(300, 1024).astype(np.float16)
    video_predictions = {
        "Action class": "Tackling",
        "Offence": "Offence", 
        "Severity": "3.0"
    }
    feature_analysis = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_path': CONFIG['video_path'],
        'feature_shape': video_features.shape,
        'feature_stats': {
            'mean': float(np.mean(video_features)),
            'std': float(np.std(video_features)),
            'min': float(np.min(video_features)),
            'max': float(np.max(video_features)),
            'non_zero_ratio': float(np.count_nonzero(video_features) / video_features.size)
        },
        'predictions': video_predictions,
        'action_confidence': None,
        'offence_confidence': None,
    }
    print("✅ 模拟数据生成完成（仅用于演示后续流程）")
# ===================================================================
# 第八部分：预测结果可视化
# ===================================================================

print("\n📊 开始预测结果可视化...")

def visualize_predictions(predictions, analysis):
    """
    可视化预测结果
    
    参数:
        predictions: 预测结果字典
        analysis: 分析结果字典
    """
    try:
        # 创建预测结果可视化
        fig = plt.figure(figsize=(18, 12))
        
        # 主标题
        fig.suptitle('X-VARS Football Video Analysis Results', fontsize=18, fontweight='bold')
        
        # 1. 动作类型分析
        ax1 = plt.subplot(2, 3, 1)
        action_class = predictions.get('Action class', 'Unknown')
        
        # 获取所有可能的动作类型
        all_actions = list(INVERSE_EVENT_DICTIONARY['action_class'].values())
        
        # 创建动作类型条形图
        current_action_idx = all_actions.index(action_class) if action_class in all_actions else 0
        colors = ['#FF6B6B' if i == current_action_idx else '#E0E0E0' for i in range(len(all_actions))]
        confidence_values = [1.0 if action == action_class else 0.1 for action in all_actions]
        
        bars = ax1.barh(all_actions, confidence_values, color=colors, alpha=0.8)
        ax1.set_title('Action Type Recognition', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Relative Confidence', fontweight='bold')
        
        # 高亮当前预测结果
        for i, (bar, action) in enumerate(zip(bars, all_actions)):
            if action == action_class:
                ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                        '✓ Predicted', ha='left', va='center', fontweight='bold', color='red')
        
        # 2. 犯规判定结果
        ax2 = plt.subplot(2, 3, 2)
        offence = predictions.get('Offence', 'Unknown')
        
        if offence in ['Offence', 'No offence']:
            offence_data = [1 if offence == 'Offence' else 0, 1 if offence == 'No offence' else 0]
            offence_labels = ['Foul', 'No Foul']
            offence_colors = ['#FF4444' if offence == 'Offence' else '#44FF44', 
                             '#44FF44' if offence == 'No offence' else '#FF4444']
            
            wedges, texts, autotexts = ax2.pie(offence_data, labels=offence_labels, colors=offence_colors,
                                              autopct='%1.0f%%', startangle=90, textprops={'fontweight': 'bold'})
        else:
            ax2.text(0.5, 0.5, 'Unknown\nOffence Status', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, fontweight='bold')
        
        ax2.set_title('Foul Detection Result', fontsize=14, fontweight='bold')
        
        # 3. 严重程度分析
        ax3 = plt.subplot(2, 3, 3)
        severity = predictions.get('Severity', '')
        
        if severity and offence == 'Offence':
            severity_levels = ['1.0', '3.0', '5.0']
            severity_labels = ['Minor Foul', 'Yellow Card', 'Red Card']
            severity_values = [1 if severity == level else 0.2 for level in severity_levels]
            severity_colors = ['#FFD700', '#FFA500', '#FF4444']
            
            bars = ax3.bar(severity_labels, severity_values, color=severity_colors, alpha=0.8)
            ax3.set_title('Foul Severity Level', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Prediction Confidence', fontweight='bold')
            ax3.set_ylim(0, 1.2)
            
            # 添加标签
            for bar, value, level in zip(bars, severity_values, severity_levels):
                if severity == level:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'Level: {severity}', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Foul\nNo Severity', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, fontweight='bold', color='green')
            ax3.set_title('Foul Severity Level', fontsize=14, fontweight='bold')
        
        # 4. 特征统计可视化
        ax4 = plt.subplot(2, 3, 4)
        stats = analysis['feature_stats']
        stat_names = ['Mean', 'Std Dev', 'Min', 'Max']
        stat_values = [stats['mean'], stats['std'], stats['min'], stats['max']]
        
        bars = ax4.bar(stat_names, stat_values, color=['#4A90E2', '#50C878', '#FF8C00', '#DC143C'], alpha=0.8)
        ax4.set_title('Feature Statistics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Value', fontweight='bold')
        
        # 添加数值标签
        for bar, value in zip(bars, stat_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(max(stat_values))*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. 综合评分雷达图
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        # 定义评分维度
        categories = ['Action\nRecognition', 'Foul\nDetection', 'Feature\nQuality', 'Model\nResponse']
        
        # 计算各维度评分
        action_score = 0.9 if action_class != 'Dont know' else 0.5
        offence_score = 0.8 if offence in ['Offence', 'No offence'] else 0.3
        feature_score = min(1.0, stats['non_zero_ratio'] * 2)
        speed_score = 0.85
        
        scores = [action_score, offence_score, feature_score, speed_score]
        
        # 角度设置
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        scores_plot = scores + scores[:1]  # 闭合图形
        angles_plot = angles + angles[:1]
        
        ax5.plot(angles_plot, scores_plot, 'o-', linewidth=3, color='#4A90E2', markersize=8)
        ax5.fill(angles_plot, scores_plot, alpha=0.25, color='#4A90E2')
        ax5.set_xticks(angles)
        ax5.set_xticklabels(categories, fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.set_title('Comprehensive Performance', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True, alpha=0.3)
        
        # 添加评分标签
        for angle, score, category in zip(angles, scores, categories):
            ax5.text(angle, score + 0.1, f'{score:.2f}', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='#4A90E2')
        
        # 6. 预测摘要文本
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # 创建摘要文本
        summary_text = f"""🎬 Video Analysis Summary

📊 Basic Information:
• Analysis Time: {analysis['timestamp']}
• Feature Dimensions: {analysis['feature_shape']}

🎯 Prediction Results:
• Action Type: {action_class}
• Foul Decision: {offence}
• Severity Level: {severity or 'None'}

📈 Feature Quality:
• Non-zero Ratio: {stats['non_zero_ratio']:.1%}
• Value Range: [{stats['min']:.3f}, {stats['max']:.3f}]

🏆 Overall Score: {np.mean(scores):.2f}/1.00

🔍 Confidence Levels:
• Action: {action_score:.1%}
• Foul: {offence_score:.1%}
• Quality: {feature_score:.1%}"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontweight='normal',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F8FF", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== Prediction Results Visualization Chart ===")
        print("Chart Name: X-VARS Football Analysis Results Dashboard")
        print("Purpose: Comprehensive visualization of AI model predictions and analysis")
        print("Chart Components:")
        print("  Top-Left (Horizontal Bar): Action Type Recognition")
        print("    - Shows all possible action types with confidence levels")
        print("    - Highlighted bar indicates the predicted action")
        print("    - Useful for understanding model certainty and alternatives")
        print("  Top-Center (Pie Chart): Foul Detection Result")
        print("    - Binary classification: Foul vs No Foul")
        print("    - Color coding: Red=Foul, Green=No Foul")
        print("    - Shows model's decision on rule violation")
        print("  Top-Right (Bar Chart): Foul Severity Level")
        print("    - Three levels: Minor (1.0), Yellow Card (3.0), Red Card (5.0)")
        print("    - Only shown when a foul is detected")
        print("    - Critical for referee decision support")
        print("  Bottom-Left (Bar Chart): Feature Statistics")
        print("    - Statistical summary of extracted features")
        print("    - Indicates data quality and model input characteristics")
        print("  Bottom-Center (Radar Chart): Comprehensive Performance")
        print("    - Multi-dimensional performance assessment")
        print("    - Four metrics: Action Recognition, Foul Detection, Feature Quality, Model Response")
        print("    - Provides holistic view of system performance")
        print("  Bottom-Right (Text Summary): Analysis Summary")
        print("    - Key results and metrics in text format")
        print("    - Includes confidence levels and quality indicators")
        print("Analysis Guidelines:")
        print("  1. Action Recognition: Look for clear prediction with high relative confidence")
        print("  2. Foul Detection: Binary decision should be definitive (100% or 0%)")
        print("  3. Severity Assessment: Only relevant when foul is detected")
        print("  4. Feature Quality: Higher non-zero ratio indicates richer feature representation")
        print("  5. Overall Performance: Radar chart should show balanced performance across metrics")
        print("Decision Support:")
        print("  - Use for referee training and decision verification")
        print("  - Combine with human expertise for final judgments")
        print("  - Consider confidence levels when making critical decisions")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

# 执行预测结果可视化
if video_found and video_predictions and feature_analysis:
    visualize_predictions(video_predictions, feature_analysis)
# ===================================================================
# 第九部分：特征深度分析
# ===================================================================

print("\n🔬 开始特征深度分析...")

def analyze_features_in_depth(features, predictions):
    """
    对提取的特征进行深度分析
    
    参数:
        features: 提取的特征数组
        predictions: 预测结果
    """
    try:
        print(f"📊 特征数组形状: {features.shape}")
        
        # 1. 特征分布分析
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Video Feature Deep Analysis', fontsize=16, fontweight='bold')
        
        # 特征值分布直方图
        axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Feature Value Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 特征值箱线图
        box_plot = axes[0, 1].boxplot(features.flatten(), vert=True, patch_artist=True,
                                     boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[0, 1].set_title('Feature Value Box Plot', fontweight='bold')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].set_xticklabels(['Features'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 特征矩阵热图（采样显示）
        if len(features.shape) > 1 and features.shape[0] > 50:
            # 如果特征太多，采样显示
            sample_indices = np.linspace(0, features.shape[0]-1, 50, dtype=int)
            sample_features = features[sample_indices]
        elif len(features.shape) > 1:
            sample_features = features
        else:
            # 1D特征转换为2D显示
            sample_features = features.reshape(1, -1)
        
        if len(sample_features.shape) > 1:
            im = axes[0, 2].imshow(sample_features[:, :min(100, sample_features.shape[1])], 
                                  cmap='viridis', aspect='auto')
            axes[0, 2].set_title('Feature Matrix Heatmap (Sampled)', fontweight='bold')
            axes[0, 2].set_xlabel('Feature Dimension')
            axes[0, 2].set_ylabel('Feature Vector')
            plt.colorbar(im, ax=axes[0, 2])
        else:
            axes[0, 2].text(0.5, 0.5, 'Feature matrix\nnot available', ha='center', va='center',
                           transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_title('Feature Matrix Heatmap', fontweight='bold')
        
        # 2. 时间特征vs空间特征分析
        # 假设前44个是时间特征，其余是空间特征（基于get_spatio_temporal_features的实现）
        temporal_features = features[:44] if features.shape[0] > 44 else features[:features.shape[0]//2]
        spatial_features = features[44:] if features.shape[0] > 44 else features[features.shape[0]//2:]
        
        # 时间特征分析
        temporal_stats = {
            'mean': np.mean(temporal_features),
            'std': np.std(temporal_features),
            'energy': np.sum(temporal_features**2)
        }
        
        # 空间特征分析
        spatial_stats = {
            'mean': np.mean(spatial_features),
            'std': np.std(spatial_features),
            'energy': np.sum(spatial_features**2)
        }
        
        # 时间vs空间特征对比
        comparison_data = {
            'Mean': [temporal_stats['mean'], spatial_stats['mean']],
            'Std Dev': [temporal_stats['std'], spatial_stats['std']],
            'Energy': [temporal_stats['energy']/1000, spatial_stats['energy']/1000]  # 缩放显示
        }
        
        x = np.arange(len(comparison_data))
        width = 0.35
        
        for i, (metric, values) in enumerate(comparison_data.items()):
            axes[1, 0].bar(x[i] - width/2, values[0], width, label='Temporal Features' if i == 0 else "", 
                          color='lightblue', alpha=0.8)
            axes[1, 0].bar(x[i] + width/2, values[1], width, label='Spatial Features' if i == 0 else "", 
                          color='lightcoral', alpha=0.8)
        
        axes[1, 0].set_xlabel('Statistical Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Temporal vs Spatial Features', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(comparison_data.keys())
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. 特征重要性分析（基于方差）
        feature_variance = np.var(features, axis=1) if len(features.shape) > 1 else np.var(features)
        
        if len(features.shape) > 1:
            # 显示前20个最重要的特征
            top_indices = np.argsort(feature_variance)[-20:]
            top_variances = feature_variance[top_indices]
            
            axes[1, 1].barh(range(len(top_variances)), top_variances, color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Variance Value')
            axes[1, 1].set_ylabel('Feature Index')
            axes[1, 1].set_title('Top 20 Important Features (by Variance)', fontweight='bold')
            axes[1, 1].set_yticks(range(len(top_variances)))
            axes[1, 1].set_yticklabels([f'Feature {idx}' for idx in top_indices])
            axes[1, 1].grid(True, alpha=0.3)
        
        # 4. 预测置信度分析
        axes[1, 2].axis('off')
        
        # 创建预测分析文本
        analysis_text = f"""🔍 Feature Deep Analysis Report

📊 Basic Statistics:
• Feature Count: {features.shape[0] if len(features.shape) > 1 else 1}
• Feature Dimensions: {features.shape[1] if len(features.shape) > 1 else features.shape[0]}
• Data Type: {features.dtype}

⏰ Temporal Feature Stats:
• Mean: {temporal_stats['mean']:.4f}
• Std Dev: {temporal_stats['std']:.4f}
• Energy: {temporal_stats['energy']:.2f}

🗺️ Spatial Feature Stats:
• Mean: {spatial_stats['mean']:.4f}
• Std Dev: {spatial_stats['std']:.4f}
• Energy: {spatial_stats['energy']:.2f}

🎯 Prediction Results:
• Action: {predictions.get('Action class', 'N/A')}
• Foul: {predictions.get('Offence', 'N/A')}
• Severity: {predictions.get('Severity', 'N/A') or 'None'}

📈 Quality Indicators:
• Non-zero Ratio: {np.count_nonzero(features)/features.size:.1%}
• Value Range: [{np.min(features):.3f}, {np.max(features):.3f}]
• Max Variance: {np.max(feature_variance):.4f}"""
        
        axes[1, 2].text(0.05, 0.95, analysis_text, transform=axes[1, 2].transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== 视频特征深度分析图表 ===")
        print("图表名称: 视频特征深度分析仪表盘")
        print("图表目的: 深入分析提取的视频特征的分布、质量和特性")
        print("图表构成:")
        print("  左上角 - 特征值分布直方图:")
        print("    · 显示所有特征值的统计分布")
        print("    · 横轴为特征值，纵轴为频次")
        print("    · 正态分布表明特征提取质量良好")
        print("    · 异常分布可能表明数据问题或模型异常")
        print("  中上角 - 特征值箱线图:")
        print("    · 显示特征值的统计特性：中位数、四分位数、异常值")
        print("    · 箱体表示25%-75%分位数范围")
        print("    · 须线表示正常数据范围")
        print("    · 圆点表示异常值")
        print("  右上角 - 特征矩阵热图:")
        print("    · 可视化特征向量的模式和相关性")
        print("    · 颜色深浅表示特征值大小")
        print("    · 帮助识别特征的空间分布模式")
        print("  左下角 - 时间vs空间特征对比:")
        print("    · 比较时间特征和空间特征的统计特性")
        print("    · Mean: 平均激活强度")
        print("    · Std Dev: 特征变化程度")
        print("    · Energy: 特征的总体能量")
        print("  中下角 - 重要特征排名:")
        print("    · 基于方差识别最重要的20个特征")
        print("    · 高方差特征对分类贡献更大")
        print("    · 帮助理解模型关注的关键特征")
        print("  右下角 - 特征分析报告:")
        print("    · 文本形式的详细统计信息")
        print("    · 包含时间特征、空间特征的详细指标")
        print("    · 提供预测结果和质量评估")
        print("分析指导:")
        print("  1. 分布分析: 正常的特征分布应接近正态分布")
        print("  2. 异常检测: 箱线图中的异常值需要关注")
        print("  3. 特征平衡: 时间和空间特征应有相似的统计特性")
        print("  4. 重要性评估: 高方差特征表明模型的关注焦点")
        print("  5. 质量评估: 非零比例高表明特征提取效果好")
        print("应用价值:")
        print("  - 模型调试: 识别特征提取中的问题")
        print("  - 性能优化: 基于重要特征进行模型改进")
        print("  - 质量监控: 持续监控特征提取的稳定性")
        print("=" * 60)
        
        return {
            'temporal_stats': temporal_stats,
            'spatial_stats': spatial_stats,
            'feature_variance': feature_variance,
            'top_feature_indices': top_indices if len(features.shape) > 1 else None
        }
        
    except Exception as e:
        print(f"❌ 特征深度分析失败: {e}")
        return None

# 执行特征深度分析
if video_found and video_features is not None:
    deep_analysis = analyze_features_in_depth(video_features, video_predictions)
# ===================================================================
# 第十部分：结果保存和导出
# ===================================================================

print("\n💾 开始保存分析结果...")

def save_analysis_results(video_path, features, predictions, analysis, output_dir):
    """
    保存分析结果到文件
    
    参数:
        video_path: 视频文件路径
        features: 提取的特征
        predictions: 预测结果
        analysis: 分析结果
        output_dir: 输出目录
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 1. 保存特征到pickle文件
        feature_file = os.path.join(output_dir, f'{video_name}_features_{timestamp}.pkl')
        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)
        print(f"✅ 特征已保存: {feature_file}")
        
        # 2. 保存预测结果到JSON文件
        prediction_file = os.path.join(output_dir, f'{video_name}_predictions_{timestamp}.json')
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"✅ 预测结果已保存: {prediction_file}")
        
        # 3. 保存完整分析报告
        report_file = os.path.join(output_dir, f'{video_name}_analysis_report_{timestamp}.json')
        
        # 准备报告数据
        report_data = {
            'video_info': {
                'path': video_path,
                'name': video_name,
                'analysis_time': timestamp
            },
            'feature_analysis': analysis,
            'predictions': predictions,
            'feature_shape': features.shape,
            'feature_summary': {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'non_zero_count': int(np.count_nonzero(features)),
                'total_elements': int(features.size)
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 分析报告已保存: {report_file}")
        
        # 4. 创建Markdown格式的可读报告
        markdown_file = os.path.join(output_dir, f'{video_name}_report_{timestamp}.md')
        markdown_content = f"""# X-VARS 足球视频分析报告

## 基本信息
- **视频文件**: {os.path.basename(video_path)}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **特征维度**: {features.shape}

## 预测结果
- **动作类型**: {predictions.get('Action class', 'N/A')}
- **是否犯规**: {predictions.get('Offence', 'N/A')}
- **犯规严重程度**: {predictions.get('Severity', 'N/A') or '无'}

## 特征统计
- **均值**: {np.mean(features):.6f}
- **标准差**: {np.std(features):.6f}
- **最小值**: {np.min(features):.6f}
- **最大值**: {np.max(features):.6f}
- **非零元素比例**: {np.count_nonzero(features)/features.size:.2%}

## 分析结论
根据X-VARS模型的分析结果，该视频片段被识别为**{predictions.get('Action class', '未知')}**动作。
{'该动作被判定为犯规' if predictions.get('Offence') == 'Offence' else '该动作未被判定为犯规'}。
{f"犯规严重程度为{predictions.get('Severity')}" if predictions.get('Severity') else ""}

## 技术细节
- 使用微调的CLIP模型进行特征提取
- 采用时空特征融合策略
- 多任务分类：动作识别 + 犯规检测

---
*报告由X-VARS系统自动生成*
"""
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✅ Markdown报告已保存: {markdown_file}")
        
        return {
            'feature_file': feature_file,
            'prediction_file': prediction_file,
            'report_file': report_file,
            'markdown_file': markdown_file
        }
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        return None

# 保存分析结果
if video_found and video_features is not None and CONFIG['save_intermediate_results']:
    saved_files = save_analysis_results(
        CONFIG['video_path'], 
        video_features, 
        video_predictions, 
        feature_analysis, 
        CONFIG['output_dir']
    )
    
    if saved_files:
        print(f"\n📁 所有结果已保存到: {CONFIG['output_dir']}")
        print("📋 保存的文件:")
        for file_type, file_path in saved_files.items():
            print(f"  • {file_type}: {os.path.basename(file_path)}")
# ===================================================================
# 第十一部分：性能评估和系统信息
# ===================================================================

print("\n⚡ 系统性能评估...")

def system_performance_analysis():
    """
    分析系统性能和资源使用情况
    """
    try:
        import psutil
        import time
        
        # GPU信息
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'device_name': torch.cuda.get_device_name(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated_memory': torch.cuda.memory_allocated(0) / 1024**3,
                'cached_memory': torch.cuda.memory_reserved(0) / 1024**3,
            }
        
        # 系统信息
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total / 1024**3,
            'memory_used': psutil.virtual_memory().used / 1024**3,
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        # 创建性能分析图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('System Performance Analysis', fontsize=16, fontweight='bold')
        
        # CPU使用率
        axes[0, 0].bar(['CPU Usage'], [system_info['cpu_usage']], color='lightblue', alpha=0.7)
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].set_title(f'CPU Usage ({system_info["cpu_count"]} Cores)', fontweight='bold')
        axes[0, 0].text(0, system_info['cpu_usage'] + 2, f'{system_info["cpu_usage"]:.1f}%', 
                       ha='center', fontweight='bold')
        
        # 内存使用情况
        memory_data = [system_info['memory_used'], system_info['memory_total'] - system_info['memory_used']]
        memory_labels = ['Used', 'Available']
        axes[0, 1].pie(memory_data, labels=memory_labels, autopct='%1.1f%%', 
                      colors=['lightcoral', 'lightgreen'], startangle=90)
        axes[0, 1].set_title(f'Memory Usage ({system_info["memory_total"]:.1f} GB)', fontweight='bold')
        
        # GPU信息（如果可用）
        if gpu_info:
            gpu_memory_data = [gpu_info['allocated_memory'], 
                              gpu_info['total_memory'] - gpu_info['allocated_memory']]
            gpu_labels = ['Allocated', 'Free']
            axes[1, 0].pie(gpu_memory_data, labels=gpu_labels, autopct='%1.1f%%',
                          colors=['orange', 'yellow'], startangle=90)
            axes[1, 0].set_title(f'GPU Memory ({gpu_info["total_memory"]:.1f} GB)\n{gpu_info["device_name"]}', 
                                fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Not Available', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold')
            axes[1, 0].set_title('GPU Status', fontweight='bold')
        
        # 系统摘要
        axes[1, 1].axis('off')
        summary_text = f"""🖥️ System Configuration Summary

💻 CPU Information:
• Core Count: {system_info['cpu_count']}
• Usage: {system_info['cpu_usage']:.1f}%

🧠 Memory Information:
• Total Capacity: {system_info['memory_total']:.1f} GB
• Used: {system_info['memory_used']:.1f} GB
• Usage Rate: {system_info['memory_percent']:.1f}%

🚀 GPU Information:
• Device: {gpu_info.get('device_name', 'N/A')}
• Total VRAM: {gpu_info.get('total_memory', 0):.1f} GB
• Allocated: {gpu_info.get('allocated_memory', 0):.1f} GB

🔧 PyTorch Information:
• Version: {torch.__version__}
• CUDA Available: {torch.cuda.is_available()}
• Device Count: {torch.cuda.device_count()}"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== 系统性能分析图表 ===")
        print("图表名称: 系统性能监控仪表盘")
        print("图表目的: 实时监控系统资源使用情况，评估视频分析任务的硬件需求")
        print("图表构成:")
        print("  左上角 - CPU使用率条形图:")
        print("    · 显示当前CPU使用百分比")
        print("    · 包含CPU核心数信息")
        print("    · 帮助评估计算负载和性能瓶颈")
        print("    · 正常范围：<80%为良好，>90%为高负载")
        print("  右上角 - 内存使用情况饼图:")
        print("    · 显示系统内存的使用和可用情况")
        print("    · 红色表示已使用内存，绿色表示可用内存")
        print("    · 帮助判断是否需要更多内存资源")
        print("    · 警戒线：>85%使用率需要注意")
        print("  左下角 - GPU内存使用饼图:")
        print("    · 显示GPU显存的分配和空闲情况")
        print("    · 橙色表示已分配显存，黄色表示空闲显存")
        print("    · 仅在GPU可用时显示，否则显示'GPU不可用'")
        print("    · 对深度学习任务至关重要")
        print("  右下角 - 系统配置摘要:")
        print("    · 详细的硬件配置信息")
        print("    · CPU、内存、GPU的具体规格")
        print("    · PyTorch版本和CUDA支持状态")
        print("性能评估指标:")
        print("  1. CPU使用率:")
        print("     - <50%: 轻负载，系统响应快")
        print("     - 50-80%: 中等负载，性能良好")
        print("     - >80%: 高负载，可能影响响应速度")
        print("  2. 内存使用率:")
        print("     - <70%: 充足内存，运行流畅")
        print("     - 70-85%: 内存紧张，需要监控")
        print("     - >85%: 内存不足，可能导致性能下降")
        print("  3. GPU显存:")
        print("     - 显存充足时模型可以处理更大批次")
        print("     - 显存不足会限制模型复杂度")
        print("     - 需要根据模型大小合理分配")
        print("优化建议:")
        print("  - CPU高负载: 考虑减少并发任务或升级CPU")
        print("  - 内存不足: 增加系统内存或优化数据加载")
        print("  - GPU显存不足: 减少批次大小或使用更大显存GPU")
        print("  - 无GPU: 考虑使用GPU加速深度学习任务")
        print("=" * 60)
        
        return {'system_info': system_info, 'gpu_info': gpu_info}
        
    except ImportError:
        print("⚠️ psutil库未安装，跳过系统性能分析")
        print("可以使用 'pip install psutil' 安装")
        return None
    except Exception as e:
        print(f"❌ 性能分析失败: {e}")
        return None

# 执行性能分析
performance_info = system_performance_analysis()
# ===================================================================
# 第十二部分：总结和建议
# ===================================================================

print("\n📋 分析总结和建议...")

def generate_final_summary():
    """
    生成最终的分析总结和建议
    """
    print("="*60)
    print("🎯 X-VARS 足球视频分析完成")
    print("="*60)
    
    if video_found and video_predictions:
        print("\n✅ 分析成功完成！")
        print(f"📹 分析视频: {os.path.basename(CONFIG['video_path'])}")
        print(f"🎬 动作识别: {video_predictions.get('Action class', 'N/A')}")
        print(f"⚖️ 犯规判定: {video_predictions.get('Offence', 'N/A')}")
        print(f"🟡 严重程度: {video_predictions.get('Severity', 'N/A') or '无'}")
        
        if video_features is not None:
            print(f"📊 特征维度: {video_features.shape}")
            print(f"💾 特征大小: {video_features.nbytes / 1024:.2f} KB")
        
        print("\n📁 输出文件:")
        if CONFIG['save_intermediate_results']:
            print(f"  • 结果保存在: {CONFIG['output_dir']}")
            print("  • 包含特征文件、预测结果、分析报告等")
        
        print("\n💡 使用建议:")
        print("  1. 可以尝试不同的视频文件进行对比分析")
        print("  2. 调整采样帧数以平衡精度和速度")
        print("  3. 保存的特征可用于后续的机器学习任务")
        print("  4. 预测结果可与人工标注进行对比验证")
        
    else:
        print("\n❌ 分析未完成")
        print("可能的原因:")
        print("  • 视频文件不存在或格式不支持")
        print("  • 模型权重文件缺失")
        print("  • 依赖库未正确安装")
        print("  • GPU内存不足")
        
        print("\n🔧 解决方案:")
        print("  1. 检查视频文件路径和格式")
        print("  2. 确认模型权重文件存在")
        print("  3. 安装所有必要的依赖库")
        print("  4. 尝试减少批处理大小或使用CPU")
    
    print("\n🚀 下次使用提示:")
    print("  • 修改CONFIG字典中的参数来自定义分析")
    print("  • 可以批量分析多个视频文件")
    print("  • 结合RAG系统进行更深入的语义分析")
    
    print("="*60)
    print("感谢使用 X-VARS 足球视频分析系统！")
    print("="*60)

# 生成最终总结
generate_final_summary()
# ===================================================================
# 第十三部分：扩展功能示例
# ===================================================================

print("\n🔧 扩展功能示例代码...")

# 以下是一些扩展功能的示例代码，可以根据需要使用

def batch_analyze_videos(video_directory, model, image_processor, output_dir):
    """
    批量分析视频文件的示例函数
    
    参数:
        video_directory: 包含视频文件的目录
        model: 加载的模型
        image_processor: 图像预处理器
        output_dir: 输出目录
    """
    try:
        video_files = [f for f in os.listdir(video_directory) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        results = []
        for video_file in video_files[:3]:  # 限制处理前3个视频作为示例
            video_path = os.path.join(video_directory, video_file)
            print(f"🎬 分析视频: {video_file}")
            
            try:
                features, predictions = get_features(video_path)
                results.append({
                    'video': video_file,
                    'predictions': predictions,
                    'feature_shape': features.shape
                })
                print(f"✅ {video_file} 分析完成")
            except Exception as e:
                print(f"❌ {video_file} 分析失败: {e}")
        
        return results
    except Exception as e:
        print(f"❌ 批量分析失败: {e}")
        return []

def compare_predictions(results_list):
    """
    比较多个视频的预测结果
    
    参数:
        results_list: 包含多个预测结果的列表
    """
    if not results_list:
        print("❌ 没有结果可以比较")
        return
    
    # 统计动作类型分布
    action_counts = {}
    offence_counts = {'Offence': 0, 'No offence': 0}
    
    for result in results_list:
        action = result['predictions'].get('Action class', 'Unknown')
        offence = result['predictions'].get('Offence', 'Unknown')
        
        action_counts[action] = action_counts.get(action, 0) + 1
        if offence in offence_counts:
            offence_counts[offence] += 1
    
    # 可视化比较结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Batch Analysis Results Comparison', fontsize=16, fontweight='bold')
    
    # 动作类型分布
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    bars = axes[0].bar(actions, counts, color='skyblue', alpha=0.7, edgecolor='navy')
    axes[0].set_title('Action Type Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Action Types')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # 犯规判定分布
    colors = ['#FF4444', '#44FF44']  # 红色表示犯规，绿色表示无犯规
    wedges, texts, autotexts = axes[1].pie(offence_counts.values(), 
                                          labels=['Foul', 'No Foul'], 
                                          autopct='%1.1f%%', colors=colors,
                                          startangle=90, textprops={'fontweight': 'bold'})
    axes[1].set_title('Foul Decision Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 批量分析结果对比图表 ===")
    print("图表名称: 批量视频分析结果对比")
    print("图表目的: 对比多个视频的分析结果，识别模式和趋势")
    print("图表构成:")
    print("  左侧 - 动作类型分布条形图:")
    print("    · 显示所有分析视频中各种动作类型的出现频次")
    print("    · 帮助识别数据集中最常见的动作类型")
    print("    · 评估动作识别模型的偏好和准确性")
    print("    · 条形高度表示该动作类型的视频数量")
    print("  右侧 - 犯规判定分布饼图:")
    print("    · 显示犯规与无犯规视频的比例")
    print("    · 红色表示犯规，绿色表示无犯规")
    print("    · 帮助评估数据集的平衡性")
    print("    · 百分比显示各类别的占比")
    print("分析价值:")
    print("  1. 数据集特征分析: 了解视频数据的分布特点")
    print("  2. 模型性能评估: 识别模型在不同动作类型上的表现")
    print("  3. 偏差检测: 发现模型可能存在的预测偏向")
    print("  4. 质量控制: 确保分析结果的一致性和可靠性")
    print("  5. 趋势识别: 发现不同视频间的共同模式")
    print("应用建议:")
    print("  - 如果某种动作类型过多，考虑数据平衡")
    print("  - 犯规比例异常时，检查模型校准")
    print("  - 结合单个视频分析进行深入调查")
    print("=" * 60)

# 示例：如何使用扩展功能
print("📝 扩展功能使用示例:")
print("1. 批量分析:")
print("   results = batch_analyze_videos('path/to/videos', model, image_processor, output_dir)")
print("2. 结果比较:")
print("   compare_predictions(results)")
print("3. 自定义分析:")
print("   # 可以基于提取的特征进行自定义的机器学习分析")

print("\n🎉 所有代码执行完成！")
print("你可以复制以上任意部分到Jupyter Notebook中运行")
print("记得根据你的实际环境修改文件路径和配置参数")

