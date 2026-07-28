'''
定义常量
'''

CONTROLLER_HEART_BEAT_EXPIRATION = 30 #控制器心跳过期时间
WORKER_HEART_BEAT_INTERVAL = 15 #工作器心跳间隔时间

LOGDIR = "." #日志目录


# Defining model
DEFAULT_VIDEO_TOKEN = "<video>" #默认视频标记
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>" #默认视频补丁标记
DEFAULT_VID_START_TOKEN = "<vid_start>" #默认视频开始标记
DEFAULT_VID_END_TOKEN = "<vid_end>" #默认视频结束标记
