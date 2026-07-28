"""
Adopted from https://github.com/gradio-app/gradio/blob/main/gradio/components.py
Fix a markdown render problem. 修复markdown渲染问题。
这是修改后的版本，适配gradio 4.26.0·
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Tuple, Union
import warnings

from gradio.components import (
    Changeable, 
    Selectable, 
    IOComponent, 
    JSONSerializable,
    Component,
    EventListenerMethod
)
from gradio import processing_utils
from markdown2 import Markdown


class _Keywords(Enum):
    """Keywords for component state management."""
    NO_VALUE = "NO_VALUE"  # 用于确定是否提供了`value`参数
    FINISHED_ITERATING = "FINISHED_ITERATING"  # 用于跳过组件的值处理（需要用于生成器+状态）


class Chatbot(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays a chatbot output showing both user submitted messages and responses. 
    Supports a subset of Markdown including bold, italics, code, and images.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects function to return a {List[Tuple[str | None | Tuple, str | None | Tuple]]}, 
    a list of tuples with user message and response messages. Messages should be strings, 
    tuples, or Nones. If the message is a string, it can include Markdown. If it is a tuple, 
    it should consist of (string filepath to image/video/audio, [optional string alt text]). 
    Messages that are `None` are not displayed.

    Demos: chatbot_simple, chatbot_multimodal 
    """
    
    def __init__(
        self,
        value: Union[List[Tuple[str, str]], Callable, None] = None,
        color_map: Union[Dict[str, str], None] = None,
        *,
        label: Union[str, None] = None,
        every: Union[float, None] = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: Union[str, None] = None,
        elem_classes: Union[List[str], str, None] = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show in chatbot. If callable, the function will be called 
                   whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while 
                   the client connection is open. Has no effect otherwise. Queue must be enabled. 
                   The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. 
                     Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component 
                         in the HTML DOM. Can be used for targeting CSS styles.
        """
        if color_map is not None:
            warnings.warn(
                "The 'color_map' parameter has been deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        
        self.md = Markdown(extras=["fenced-code-blocks", "tables", "break-on-newline"])
        self.select: EventListenerMethod
        """
        Event listener for when the user selects message from Chatbot.
        Uses event data gradio.SelectData to carry `value` referring to text of selected message, 
        and `index` tuple to refer to [message, participant] index.
        See EventData documentation on how to use this event data.
        """

        super().__init__(
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        """Get component configuration."""
        return {
            "value": self.value,
            "selectable": self.selectable,
            **super().get_config(),
        }

    @staticmethod
    def update(
        value: Union[Any, Literal[_Keywords.NO_VALUE], None] = _Keywords.NO_VALUE,
        label: Union[str, None] = None,
        show_label: Union[bool, None] = None,
        visible: Union[bool, None] = None,
    ):
        """Update component configuration."""
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def _process_chat_messages(
        self, 
        chat_message: Union[str, Tuple, List, Dict, None]
    ) -> Union[str, Dict, None]:
        """Process individual chat messages."""
        if chat_message is None:
            return None
        elif isinstance(chat_message, (tuple, list)):
            mime_type = processing_utils.get_mimetype(chat_message[0])
            return {
                "name": chat_message[0],
                "mime_type": mime_type,
                "alt_text": chat_message[1] if len(chat_message) > 1 else None,
                "data": None,  # 这些最后两个字段由前端填充
                "is_file": True,
            }
        elif isinstance(chat_message, dict):
            # 这是之前处理过的消息
            return chat_message
        elif isinstance(chat_message, str):
            # 返回渲染后的消息
            return str(self.md.convert(chat_message))
        else:
            raise ValueError(f"Invalid message for Chatbot component: {chat_message}")

    def postprocess(
        self,
        y: List[Tuple[Union[str, Tuple, List, Dict, None], Union[str, Tuple, List, Dict, None]]],
    ) -> List[Tuple[Union[str, Dict, None], Union[str, Dict, None]]]:
        """
        Parameters:
            y: List of tuples representing the message and response pairs. Each message and response 
               should be a string, which may be in Markdown format. It can also be a tuple whose 
               first element is a string filepath or URL to an image/video/audio, and second 
               (optional) element is the alt text, in which case the media file is displayed. 
               It can also be None, in which case that message is not displayed.
        Returns:
            List of tuples representing the message and response. Each message and response will be 
            a string of HTML, or a dictionary with media information.
        """
        if y is None:
            return []
        
        processed_messages = []
        for message_pair in y:
            assert isinstance(
                message_pair, (tuple, list)
            ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
            assert (
                len(message_pair) == 2
            ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"
            
            processed_messages.append(
                (
                    # 用户消息使用 pre 标签包装，保持原始格式
                    '<pre style="font-family: var(--font)">' +
                    str(message_pair[0]) + "</pre>",
                    # 助手消息使用 markdown 渲染
                    self._process_chat_messages(message_pair[1]),
                )
            )
        return processed_messages

    def style(self, height: Union[int, None] = None, **kwargs):
        """
        This method can be used to change the appearance of the Chatbot component.
        """
        if height is not None:
            self._style["height"] = height
        if kwargs.get("color_map") is not None:
            warnings.warn(
                "The 'color_map' parameter has been deprecated.",
                DeprecationWarning,
                stacklevel=2
            )

        super().style(**kwargs)
        return self


