// ⚽ X-VARS 智能足球裁判系统 - 足球主题交互脚本 ⚽

(function() {
    'use strict';
    
    // 等待DOM加载完成
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    function init() {
        // 添加足球场装饰元素
        addFootballFieldDecorations();
        
        // 美化视频容器
        enhanceVideoContainer();
        
        // 添加消息动画
        enhanceChatMessages();
        
        // 添加按钮点击效果
        enhanceButtons();
        
        // 添加加载提示
        addLoadingAnimations();
        
        // 添加人性化提示
        addHelpfulTips();
        
        // 监听新消息
        observeNewMessages();
    }
    
    // 添加足球场装饰
    function addFootballFieldDecorations() {
        // 在视频区域添加足球场线条
        const videoContainers = document.querySelectorAll('[data-testid="stVideo"]');
        videoContainers.forEach(container => {
            if (!container.classList.contains('football-enhanced')) {
                container.classList.add('football-enhanced');
                
                // 创建装饰容器
                const decor = document.createElement('div');
                decor.className = 'football-field-decor';
                decor.innerHTML = `
                    <div class="field-line center-line"></div>
                    <div class="field-line penalty-left"></div>
                    <div class="field-line penalty-right"></div>
                    <div class="field-circle center-circle"></div>
                `;
                
                // 插入装饰
                container.parentElement.insertBefore(decor, container);
            }
        });
    }
    
    // 美化视频容器
    function enhanceVideoContainer() {
        const videoElements = document.querySelectorAll('video');
        videoElements.forEach(video => {
                // 移除视频播放提示（简约风格不需要）
            if (!video.hasAttribute('data-enhanced')) {
                video.setAttribute('data-enhanced', 'true');
                // 不添加任何事件监听器，保持简洁
            }
        });
    }
    
    // 美化聊天消息
    function enhanceChatMessages() {
        const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
        chatMessages.forEach((message, index) => {
            if (!message.hasAttribute('data-animated')) {
                message.setAttribute('data-animated', 'true');
                
            // 添加进入动画（简化版，无延迟）
            message.style.opacity = '0';
            message.style.transform = 'translateY(10px)';
            
            setTimeout(() => {
                message.style.transition = 'all 0.3s ease-out';
                message.style.opacity = '1';
                message.style.transform = 'translateY(0)';
            }, 50);
                
                // 添加点击复制功能
                message.addEventListener('click', function() {
                    const text = message.innerText;
                    if (text && navigator.clipboard) {
                        navigator.clipboard.writeText(text).then(() => {
                            showNotification('📋 消息已复制到剪贴板', 'success');
                        });
                    }
                });
                
                // 添加悬停效果
                message.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.02)';
                    this.style.transition = 'transform 0.2s ease';
                });
                
                message.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                });
            }
        });
    }
    
    // 美化按钮
    function enhanceButtons() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            if (!button.hasAttribute('data-enhanced')) {
                button.setAttribute('data-enhanced', 'true');
                
                // 添加点击波纹效果
                button.addEventListener('click', function(e) {
                    const ripple = document.createElement('span');
                    ripple.className = 'ripple-effect';
                    
                    const rect = this.getBoundingClientRect();
                    const size = Math.max(rect.width, rect.height);
                    const x = e.clientX - rect.left - size / 2;
                    const y = e.clientY - rect.top - size / 2;
                    
                    ripple.style.width = ripple.style.height = size + 'px';
                    ripple.style.left = x + 'px';
                    ripple.style.top = y + 'px';
                    
                    this.appendChild(ripple);
                    
                    setTimeout(() => {
                        ripple.remove();
                    }, 600);
                });
            }
        });
    }
    
    // 添加加载动画
    function addLoadingAnimations() {
        // 查找加载指示器
        const spinners = document.querySelectorAll('[data-testid="stSpinner"]');
        spinners.forEach(spinner => {
            if (!spinner.hasAttribute('data-enhanced')) {
                spinner.setAttribute('data-enhanced', 'true');
                
                // 添加加载文本
                const loadingText = document.createElement('div');
                loadingText.className = 'loading-text';
                loadingText.textContent = '⚽ 正在分析视频...';
                spinner.appendChild(loadingText);
            }
        });
    }
    
    // 移除人性化提示（简约风格不需要）
    function addHelpfulTips() {
        // 简约风格不显示提示卡片
        return;
    }
    
    // 监听新消息
    function observeNewMessages() {
        const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
        if (!chatContainer) return;
        
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    // 新消息添加，重新美化
                    setTimeout(() => {
                        enhanceChatMessages();
                    }, 100);
                }
            });
        });
        
        observer.observe(chatContainer, {
            childList: true,
            subtree: true
        });
    }
    
    // 移除通知功能（简约风格不需要）
    function showNotification(message, type = 'info') {
        // 简约风格不显示通知
        return;
    }
    
    // 添加CSS样式（如果CSS文件未加载）
    function injectStyles() {
        if (document.getElementById('football-theme-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'football-theme-styles';
        style.textContent = `
            .football-field-decor {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                pointer-events: none;
                z-index: 1;
            }
            
            .field-line {
                position: absolute;
                background: rgba(255,255,255,0.2);
                border: 1px dashed rgba(255,255,255,0.3);
            }
            
            .field-line.center-line {
                top: 50%;
                left: 0;
                right: 0;
                height: 2px;
                transform: translateY(-50%);
            }
            
            .field-circle {
                position: absolute;
                top: 50%;
                left: 50%;
                width: 100px;
                height: 100px;
                border: 2px dashed rgba(255,255,255,0.3);
                border-radius: 50%;
                transform: translate(-50%, -50%);
            }
            
            .ripple-effect {
                position: absolute;
                border-radius: 50%;
                background: rgba(255,255,255,0.6);
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            }
            
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
            
            .football-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 100%);
                color: white;
                border: 2px solid #ffd700;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                z-index: 10000;
                transform: translateX(400px);
                transition: transform 0.3s ease;
                font-weight: bold;
            }
            
            .football-notification.show {
                transform: translateX(0);
            }
            
            .football-notification.success {
                border-color: #28a745;
            }
            
            .football-notification.info {
                border-color: #007bff;
            }
            
            .football-tips-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: linear-gradient(135deg, rgba(45,80,22,0.95) 0%, rgba(61,107,31,0.95) 100%);
                border: 3px solid #ffd700;
                border-radius: 15px;
                padding: 20px;
                max-width: 300px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4);
                z-index: 9999;
                animation: slideInUp 0.5s ease-out;
            }
            
            @keyframes slideInUp {
                from {
                    transform: translateY(100px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            .tip-card {
                color: white;
            }
            
            .tip-card h4 {
                margin: 0 0 10px 0;
                color: #ffd700;
            }
            
            .tip-card ul {
                margin: 0;
                padding-left: 20px;
            }
            
            .tip-card li {
                margin: 5px 0;
            }
            
            .tip-close-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: transparent;
                border: none;
                color: white;
                font-size: 24px;
                cursor: pointer;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: all 0.3s ease;
            }
            
            .tip-close-btn:hover {
                background: rgba(255,255,255,0.2);
                transform: rotate(90deg);
            }
        `;
        
        document.head.appendChild(style);
    }
    
    // 注入样式
    injectStyles();
    
})();

