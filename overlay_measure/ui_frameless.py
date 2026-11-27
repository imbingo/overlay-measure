from __future__ import annotations

import sys


class FramelessWindowMixin:
    """Restore native edge and corner resizing for a frameless Windows window."""

    _WM_NCHITTEST = 0x0084
    _HTCLIENT = 1
    _HTLEFT = 10
    _HTRIGHT = 11
    _HTTOP = 12
    _HTTOPLEFT = 13
    _HTTOPRIGHT = 14
    _HTBOTTOM = 15
    _HTBOTTOMLEFT = 16
    _HTBOTTOMRIGHT = 17
    _RESIZE_MARGIN_DIP = 8

    @classmethod
    def _frameless_hit_test(
        cls,
        x: int,
        y: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
        margin: int,
    ) -> int:
        """Return the Windows non-client hit code for a frameless window edge."""
        near_left = left <= x < left + margin
        near_right = right - margin <= x < right
        near_top = top <= y < top + margin
        near_bottom = bottom - margin <= y < bottom

        if near_top and near_left:
            return cls._HTTOPLEFT
        if near_top and near_right:
            return cls._HTTOPRIGHT
        if near_bottom and near_left:
            return cls._HTBOTTOMLEFT
        if near_bottom and near_right:
            return cls._HTBOTTOMRIGHT
        if near_left:
            return cls._HTLEFT
        if near_right:
            return cls._HTRIGHT
        if near_top:
            return cls._HTTOP
        if near_bottom:
            return cls._HTBOTTOM
        return cls._HTCLIENT

    def nativeEvent(self, event_type, message):
        if sys.platform == "win32" and not self.isMaximized() and not self.isFullScreen():
            if bytes(event_type) != b"windows_generic_MSG":
                return super().nativeEvent(event_type, message)
            import ctypes
            from ctypes import wintypes

            get_window_rect = ctypes.windll.user32.GetWindowRect
            if not getattr(get_window_rect, "argtypes", None):
                get_window_rect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
                get_window_rect.restype = wintypes.BOOL

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == self._WM_NCHITTEST:
                rect = wintypes.RECT()
                if get_window_rect(wintypes.HWND(int(self.winId())), ctypes.byref(rect)):
                    packed_position = int(msg.lParam)
                    x = ctypes.c_short(packed_position & 0xFFFF).value
                    y = ctypes.c_short((packed_position >> 16) & 0xFFFF).value
                    margin = max(6, round(self._RESIZE_MARGIN_DIP * self.devicePixelRatioF()))
                    hit = self._frameless_hit_test(
                        x,
                        y,
                        rect.left,
                        rect.top,
                        rect.right,
                        rect.bottom,
                        margin,
                    )
                    if hit != self._HTCLIENT:
                        return True, hit
        return super().nativeEvent(event_type, message)
