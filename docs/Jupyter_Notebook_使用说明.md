# X-VARS Jupyter Notebook 视频分析使用说明

## 📋 概述

这个Jupyter Notebook代码提供了完整的足球视频分析功能，基于X-VARS系统进行视频特征提取、动作识别和犯规检测。

## 🚀 快速开始

### 1. 环境准备

确保已安装以下依赖：
```bash
pip install torch torchvision transformers
pip install matplotlib seaborn pillow
pip install decord numpy opencv-python
pip install psutil  # 可选，用于系统性能监控
```

### 2. 使用步骤

1. **复制代码**：将 `video_analysis_notebook.py` 的内容复制到Jupyter Notebook中

2. **修改配置**：根据你的环境修改以下参数：
```python
CONFIG = {
    # 修改为你的视频文件路径
    'video_path': r"E:\your_project\dataset\videos\clip_1.mp4",
    
    # 修改为你的模型权重路径
    'model_weights': r"E:\your_project\visual_encoder\14_model.pth.tar",
    
    # 其他配置保持默认即可
    'output_dir': r"E:\your_project\analysis_results",
}
```

3. **逐步执行**：按照代码中的13个部分逐个执行

## 📊 主要功能模块

### 第一部分：环境设置
- 导入必要的库
- 检查PyTorch和CUDA状态
- 设置中文字体支持

### 第二部分：路径配置
- 设置项目路径
- 导入自定义模块
- 检查依赖是否正确

### 第三部分：参数配置
- 配置视频文件路径
- 设置模型参数
- 创建输出目录

### 第四部分：模型初始化
- 加载CLIP预处理器
- 初始化CLIPNetwork模型
- 加载预训练权重

### 第五部分：视频信息分析
- 提取视频基本信息
- 显示文件大小、帧率、分辨率等
- 生成信息可视化图表

### 第六部分：帧采样预览
- 智能帧采样
- 生成视频帧预览图
- 显示关键帧内容

### 第七部分：特征提取
- 使用X-VARS模型提取特征
- 进行动作识别和犯规检测
- 输出预测结果

### 第八部分：结果可视化
- 预测结果图表展示
- 动作类型、犯规判定可视化
- 综合性能雷达图

### 第九部分：特征深度分析
- 特征分布统计
- 时间vs空间特征对比
- 特征重要性分析

### 第十部分：结果保存
- 保存特征到.pkl文件
- 导出预测结果到.json
- 生成分析报告

### 第十一部分：性能评估
- 系统资源使用情况
- GPU内存占用分析
- 性能指标可视化

### 第十二部分：总结建议
- 分析结果总结
- 使用建议和提示

### 第十三部分：扩展功能
- 批量视频分析示例
- 结果对比分析
- 自定义扩展代码

## 📈 输出结果

### 可视化图表
- 视频基本信息分析图
- 视频帧预览图
- 预测结果可视化
- 特征深度分析图
- 系统性能分析图

### 保存文件
- `{video_name}_features_{timestamp}.pkl` - 特征数据
- `{video_name}_predictions_{timestamp}.json` - 预测结果
- `{video_name}_analysis_report_{timestamp}.json` - 详细报告
- `{video_name}_report_{timestamp}.md` - Markdown报告

## 🔧 常见问题

### Q1: 模块导入失败
**A**: 检查项目路径设置是否正确，确保 `visual_encoder` 目录在Python搜索路径中

### Q2: 视频文件不存在
**A**: 修改 `CONFIG['video_path']` 为正确的视频文件路径

### Q3: GPU内存不足
**A**: 
- 减少 `num_frames_to_analyze` 参数
- 或者修改代码使用CPU：将 `.cuda()` 改为 `.cpu()`

### Q4: 模型权重文件缺失
**A**: 确保 `14_model.pth.tar` 文件存在，或修改 `CONFIG['model_weights']` 路径

## 💡 使用技巧

1. **首次运行**：建议先运行前4个部分确保环境正确
2. **内存优化**：如果内存不足，可以分段执行代码
3. **结果保存**：设置 `save_intermediate_results=True` 保存所有中间结果
4. **批量分析**：使用第十三部分的扩展功能处理多个视频

## 📞 技术支持

如果遇到问题，请检查：
1. Python环境和依赖库版本
2. 文件路径是否正确
3. 模型文件是否完整
4. 系统资源是否充足

## 🎯 预期结果

成功运行后，你将获得：
- 视频的动作类型识别结果
- 犯规判定和严重程度评估
- 详细的特征分析报告
- 完整的可视化图表
- 可用于后续分析的特征数据

---

*本工具基于X-VARS (Explainable Video Assistant Referee System) 开发*
