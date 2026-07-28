"""
Adopted from https://github.com/gradio-app/gradio/blob/main/gradio/components.py
Fix a markdown render problem.
这是原始版
"""
from __future__ import annotations #兼容性

from gradio.components import *
from markdown2 import Markdown


class _Keywords(Enum):
    NO_VALUE = "NO_VALUE"  # 用于确定是否提供了`value`参数
    FINISHED_ITERATING = "FINISHED_ITERATING"  # 用于跳过组件的值处理（需要用于生成器+状态）


@document("style") #文档
class Chatbot(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays a chatbot output showing both user submitted messages and responses. Supports a subset of Markdown including bold, italics, code, and images.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects function to return a {List[Tuple[str | None | Tuple, str | None | Tuple]]}, a list of tuples with user message and response messages. Messages should be strings, tuples, or Nones. If the message is a string, it can include Markdown. If it is a tuple, it should consist of (string filepath to image/video/audio, [optional string alt text]). Messages that are `None` are not displayed.

    Demos: chatbot_simple, chatbot_multimodal
    """

    def __init__(
        self,
        value: List[Tuple[str | None, str | None]] | Callable | None = None, #值
        color_map: Dict[str, str] | None = None,  # 参数移动到Chatbot.style()
        *,
        label: str | None = None, #标签
        every: float | None = None, #每隔多少秒
        show_label: bool = True, #是否显示标签
        visible: bool = True, #是否可见
        elem_id: str | None = None, #元素id
        elem_classes: List[str] | str | None = None, #元素类
        **kwargs, #其他参数
    ):
        """
        Parameters:
            value: Default value to show in chatbot. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        if color_map is not None: #如果颜色映射不为空
            warnings.warn( #警告
                "The 'color_map' parameter has been deprecated.", #警告信息
            )
        #self.md = utils.get_markdown_parser() #获取markdown解析器
        self.md = Markdown(extras=["fenced-code-blocks", "tables", "break-on-newline"]) #markdown解析器
        self.select: EventListenerMethod #事件监听器
        """
        Event listener for when the user selects message from Chatbot.
        Uses event data gradio.SelectData to carry `value` referring to text of selected message, and `index` tuple to refer to [message, participant] index.
        See EventData documentation on how to use this event data.
        """

        IOComponent.__init__( #初始化
            self, #初始化
            label=label, #标签
            every=every, #每隔多少秒
            show_label=show_label, #是否显示标签
            visible=visible, #是否可见
            elem_id=elem_id, #元素id
            elem_classes=elem_classes, #元素类
            value=value, #值
            **kwargs, #其他参数
        )

    def get_config(self): #获取配置
        return { #返回配置
            "value": self.value, #值
            "selectable": self.selectable, #是否可选择
            **IOComponent.get_config(self), #其他配置
        }

    @staticmethod
    def update( #更新
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE, #值
        label: str | None = None, #标签
        show_label: bool | None = None, #是否显示标签
        visible: bool | None = None, #是否可见
    ):
        updated_config = { #更新配置
            "label": label, #标签
            "show_label": show_label, #是否显示标签
            "visible": visible, #是否可见
            "value": value, #值
            "__type__": "update", #类型
        }
        return updated_config #返回更新配置

    def _process_chat_messages( #处理聊天消息
        self, chat_message: str | Tuple | List | Dict | None #聊天消息
    ) -> str | Dict | None:
        if chat_message is None: #如果聊天消息为空
            return None #返回空
        elif isinstance(chat_message, (tuple, list)): #如果聊天消息为元组或列表
            mime_type = processing_utils.get_mimetype(chat_message[0]) #获取mime类型
            return { #返回消息
                "name": chat_message[0], #名称
                "mime_type": mime_type, #mime类型
                "alt_text": chat_message[1] if len(chat_message) > 1 else None, #alt文本
                "data": None,  # 这些最后两个字段由前端填充
                "is_file": True, #是否为文件
            }
        elif isinstance(chat_message, dict):  # 这是之前处理过的消息
            return chat_message
        elif isinstance(chat_message, str): #如果聊天消息为字符串
            #return self.md.render(chat_message) #返回渲染后的消息
            return str(self.md.convert(chat_message)) #返回渲染后的消息
        else: #如果聊天消息为其他类型
            raise ValueError(f"Invalid message for Chatbot component: {chat_message}") #抛出错误

    def postprocess( #处理消息
        self,
        y: List[ #消息列表
            Tuple[str | Tuple | List | Dict | None, str | Tuple | List | Dict | None] #消息元组
        ],
    ) -> List[Tuple[str | Dict | None, str | Dict | None]]: #返回消息列表
        """
        Parameters:
            y: List of tuples representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.  It can also be a tuple whose first element is a string filepath or URL to an image/video/audio, and second (optional) element is the alt text, in which case the media file is displayed. It can also be None, in which case that message is not displayed.
        Returns:
            List of tuples representing the message and response. Each message and response will be a string of HTML, or a dictionary with media information.
        """
        if y is None: #如果消息为空
            return [] #返回空列表
        processed_messages = [] #处理后的消息列表
        for message_pair in y: #遍历消息列表
            assert isinstance(message_pair, (tuple, list)), f"Expected a list of lists or list of tuples. Received: {message_pair}" #如果消息为元组或列表
            assert len(message_pair) == 2, f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}" #如果消息长度为2
            processed_messages.append( #添加消息
                (
                    #self._process_chat_messages(message_pair[0]), #处理消息
                    '<pre style="font-family: var(--font)">' + message_pair[0] + "</pre>", #添加消息
                    self._process_chat_messages(message_pair[1]), #处理消息
                )
            )
        return processed_messages

    def style(self, height: int | None = None, **kwargs): #样式
        """
        This method can be used to change the appearance of the Chatbot component.
        """
        if height is not None: #如果高度不为空
            self._style["height"] = height #设置高度
        if kwargs.get("color_map") is not None: #如果颜色映射不为空
            warnings.warn("The 'color_map' parameter has been deprecated.") #警告

        Component.style( #样式
            self, #样式
            **kwargs, #其他参数
        )
        return self #返回样式


