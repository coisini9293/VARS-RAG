# 足球主题美化文件

本目录包含X-VARS智能足球裁判系统的UI美化文件。

## 文件说明

- `football_theme.css` - 足球主题CSS样式文件
  - 包含所有UI元素的样式定义
  - 足球场绿色背景
  - 按钮、对话框、视频容器等美化样式
  
- `football_theme.js` - 足球主题JavaScript交互脚本
  - 消息动画效果
  - 按钮点击效果
  - 人性化提示
  - 视频播放状态通知

## 使用方法

这些文件已自动集成到主程序中，无需手动加载。

运行 `streamlit run x_vars_demo_with_streamlit.py` 时，程序会自动：
1. 检查 `static/football_theme.css` 是否存在
2. 如果存在，加载CSS样式
3. 如果不存在，使用内联简化版CSS
4. 同样处理JavaScript文件

## 自定义美化

### 修改颜色主题

编辑 `football_theme.css` 文件，修改CSS变量：

```css
:root {
    --football-green: #2d5016;        /* 主绿色 */
    --football-light-green: #4a7c2a;   /* 浅绿色 */
    --football-field-green: #3d6b1f;   /* 场地绿色 */
    --football-yellow: #ffd700;        /* 金黄色 */
    --football-white: #ffffff;         /* 白色 */
    --football-red: #dc3545;           /* 红色 */
    --football-blue: #007bff;          /* 蓝色 */
}
```

### 修改动画效果

编辑 `football_theme.js` 文件，调整动画参数：

```javascript
// 消息进入动画延迟
setTimeout(() => {
    message.style.transition = 'all 0.5s ease-out';  // 修改动画时长
    message.style.opacity = '1';
    message.style.transform = 'translateY(0)';
}, index * 100);  // 修改延迟时间
```

### 添加新功能

在 `football_theme.js` 的 `init()` 函数中添加新的初始化函数调用。

## 注意事项

- 修改CSS/JS后需要刷新浏览器才能看到效果
- 建议使用浏览器开发者工具调试样式
- 保持文件编码为UTF-8
- 确保CSS选择器与Streamlit的DOM结构匹配

