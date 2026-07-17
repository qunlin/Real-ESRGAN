#!/usr/bin/env python
"""
Real-ESRGAN GUI Application
A tkinter-based GUI for image super-resolution using Real-ESRGAN.
Supports both standalone Python execution and PyInstaller-packaged .exe.

Features:
  - Select input image (with drag-and-drop support)
  - Preview input & output images side-by-side
  - Choose from 6 built-in models (auto-download weights)
  - One-click enhance, then download/save the result
  - Cross-platform: Windows, macOS, Linux
"""

import os
import sys
import threading
import traceback
from pathlib import Path

# ── PyInstaller / frozen app path resolution ──────────────────────────
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys._MEIPASS)
else:
    APP_DIR = Path(__file__).resolve().parent

WEIGHTS_DIR = APP_DIR / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── GUI imports ────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# ── Image processing imports ───────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageTk
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


# ══════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════

MODELS = {
    'RealESRGAN_x4plus': {
        'name': 'RealESRGAN x4plus (通用照片 4x)',
        'arch': 'rrdbnet',
        'scale': 4,
        'num_block': 23,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    },
    'RealESRGAN_x2plus': {
        'name': 'RealESRGAN x2plus (通用照片 2x)',
        'arch': 'rrdbnet',
        'scale': 2,
        'num_block': 23,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    },
    'RealESRGAN_x4plus_anime_6B': {
        'name': 'RealESRGAN x4plus Anime (动漫 4x)',
        'arch': 'rrdbnet',
        'scale': 4,
        'num_block': 6,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    },
    'RealESRNet_x4plus': {
        'name': 'RealESRNet x4plus (通用 4x, 无GAN)',
        'arch': 'rrdbnet',
        'scale': 4,
        'num_block': 23,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    },
    'realesr-animevideov3': {
        'name': 'RealESRGAN AnimeVideo v3 (动漫视频)',
        'arch': 'srvgg',
        'scale': 4,
        'num_conv': 16,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
    },
    'realesr-general-x4v3': {
        'name': 'RealESRGAN General v3 (轻量通用)',
        'arch': 'srvgg',
        'scale': 4,
        'num_conv': 32,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    },
}

# Supported image formats
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def get_model_path(model_key):
    """Return local model path, downloading if necessary."""
    info = MODELS[model_key]
    local_path = WEIGHTS_DIR / f'{model_key}.pth'

    if local_path.exists():
        return str(local_path)

    log(f'正在下载模型 {model_key} ...')
    try:
        downloaded = load_file_from_url(
            url=info['url'],
            model_dir=str(WEIGHTS_DIR),
            progress=True,
            file_name=None,
        )
        log(f'模型已下载到: {downloaded}')
        return downloaded
    except Exception as e:
        log(f'下载失败: {e}')
        raise


def create_upsampler(model_key, tile, tile_pad, pre_pad, fp32, gpu_id):
    """Build and return a RealESRGANer instance for the given model."""
    info = MODELS[model_key]
    model_path = get_model_path(model_key)

    if info['arch'] == 'rrdbnet':
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=info['num_block'],
            num_grow_ch=32, scale=info['scale'],
        )
    else:
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=info['num_conv'],
            upscale=info['scale'], act_type='prelu',
        )

    upsampler = RealESRGANer(
        scale=info['scale'],
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id,
    )
    return upsampler, info['scale']


# ══════════════════════════════════════════════════════════════════════════
# Logging to GUI
# ══════════════════════════════════════════════════════════════════════════

_log_widget = None

def set_log_widget(widget):
    global _log_widget
    _log_widget = widget

def log(msg):
    """Print to both console and GUI log area."""
    print(msg)
    if _log_widget is not None:
        _log_widget.insert(tk.END, msg + '\n')
        _log_widget.see(tk.END)
        _log_widget.update_idletasks()


# ══════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════

class App:
    PREVIEW_MAX_W = 350
    PREVIEW_MAX_H = 280

    def __init__(self, root):
        self.root = root
        self.root.title('Real-ESRGAN 图像超分辨率 v1.0')
        self.root.geometry('960x750')
        self.root.minsize(800, 650)

        # State
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / 'Desktop' / 'Real-ESRGAN-Output'))
        self.model_key = tk.StringVar(value='RealESRGAN_x4plus')
        self.tile_size = tk.IntVar(value=512)
        self.outscale = tk.DoubleVar(value=4.0)
        self.fp32 = tk.BooleanVar(value=True)
        self.suffix = tk.StringVar(value='out')
        self.processing = False
        self.result_path = None
        self._input_pil = None    # original PIL image
        self._output_pil = None   # enhanced PIL image

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        pad = {'padx': 8, 'pady': 4}

        # ── Outer container: left (control) | right (preview) ─────────
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left_panel = ttk.Frame(outer, width=480)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(outer, width=440)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ── LEFT: Controls ────────────────────────────────────────────

        # Input image
        in_frame = ttk.LabelFrame(left_panel, text='📁 输入图片')
        in_frame.pack(fill=tk.X, **pad)

        r0 = ttk.Frame(in_frame)
        r0.pack(fill=tk.X, **pad)
        self.input_entry = ttk.Entry(r0, textvariable=self.input_path)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r0, text='选择图片...', command=self._browse_input).pack(side=tk.LEFT, padx=(4, 0))
        # Hint
        if HAS_DND:
            ttk.Label(in_frame, text='💡 也可将图片文件直接拖拽到窗口',
                      foreground='gray').pack(anchor=tk.W, padx=12, pady=(0, 4))

        # Model selector
        md_frame = ttk.LabelFrame(left_panel, text='🧠 模型选择')
        md_frame.pack(fill=tk.X, **pad)

        r1 = ttk.Frame(md_frame)
        r1.pack(fill=tk.X, **pad)
        model_names = [info['name'] for info in MODELS.values()]
        self.model_combo = ttk.Combobox(r1, values=model_names, state='readonly', width=50)
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)

        # Output directory
        out_frame = ttk.LabelFrame(left_panel, text='💾 输出目录 (照片保存位置)')
        out_frame.pack(fill=tk.X, **pad)

        r2 = ttk.Frame(out_frame)
        r2.pack(fill=tk.X, **pad)
        ttk.Entry(r2, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r2, text='选择目录...', command=self._browse_output).pack(side=tk.LEFT, padx=(4, 0))

        # Options
        opt_frame = ttk.LabelFrame(left_panel, text='⚙️ 处理选项')
        opt_frame.pack(fill=tk.X, **pad)

        r3 = ttk.Frame(opt_frame)
        r3.pack(fill=tk.X, **pad)

        ttk.Label(r3, text='分块大小:').pack(side=tk.LEFT)
        ttk.Spinbox(r3, textvariable=self.tile_size, from_=0, to=2048,
                     increment=128, width=6).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(r3, text='放大倍数:').pack(side=tk.LEFT)
        ttk.Spinbox(r3, textvariable=self.outscale, from_=1.0, to=8.0,
                     increment=0.5, width=5).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Checkbutton(r3, text='FP32', variable=self.fp32).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(r3, text='后缀:').pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.suffix, width=8).pack(side=tk.LEFT, padx=(2, 0))

        # Action buttons
        act_frame = ttk.Frame(left_panel)
        act_frame.pack(fill=tk.X, **pad)

        self.process_btn = ttk.Button(act_frame, text='🚀 开始处理', command=self._start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.download_btn = ttk.Button(act_frame, text='💾 另存为...', command=self._save_as, state=tk.DISABLED)
        self.download_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.open_btn = ttk.Button(act_frame, text='📂 打开输出目录', command=self._open_output_dir, state=tk.DISABLED)
        self.open_btn.pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, **pad)

        # Status
        self.status_var = tk.StringVar(value='就绪 — 选择图片后点击"开始处理"')
        status_label = ttk.Label(left_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=8, pady=(0, 4))

        # Result path
        res_frame = ttk.LabelFrame(left_panel, text='✅ 处理结果')
        res_frame.pack(fill=tk.X, **pad)

        self.result_var = tk.StringVar(value='（等待处理）')
        res_row = ttk.Frame(res_frame)
        res_row.pack(fill=tk.X, **pad)
        self.result_entry = ttk.Entry(res_row, textvariable=self.result_var, state='readonly')
        self.result_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Log
        log_frame = ttk.LabelFrame(left_panel, text='📋 运行日志')
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD, state=tk.NORMAL)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)

        set_log_widget(self.log_text)

        # ── RIGHT: Preview ────────────────────────────────────────────
        prev_frame = ttk.LabelFrame(right_panel, text='🖼️ 图片预览')
        prev_frame.pack(fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)

        # Before label
        self._before_label = ttk.Label(prev_frame, text='原图', anchor=tk.CENTER)
        self._before_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # After label
        self._after_label = ttk.Label(prev_frame, text='增强后', anchor=tk.CENTER)
        self._after_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Placeholder canvases
        self._before_canvas = self._make_preview_canvas(prev_frame, '暂无图片\n\n将图片拖拽到此处\n或点击左侧"选择图片"')
        self._before_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self._after_canvas = self._make_preview_canvas(prev_frame, '处理后将显示\n增强后的图片')
        self._after_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Drag-and-drop registration
        self._setup_drop_targets()

        # Init
        log('Real-ESRGAN GUI 已启动。')
        log(f'默认模型: RealESRGAN_x4plus')
        log(f'输出目录: {self.output_dir.get()}')

    def _make_preview_canvas(self, parent, placeholder_text):
        """Create a bordered canvas for preview images."""
        frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas = tk.Canvas(frame, bg='#f0f0f0', highlightthickness=0, width=200, height=200)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.placeholder = placeholder_text
        canvas._pil_image = None
        # Draw placeholder text
        canvas.bind('<Configure>', lambda e, c=canvas: self._draw_placeholder(c))
        return frame

    def _draw_placeholder(self, canvas):
        canvas.delete('placeholder')
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w > 10 and h > 10:
            canvas.create_text(
                w // 2, h // 2, text=canvas.placeholder,
                fill='#888888', font=('', 11), justify=tk.CENTER,
                tags='placeholder',
            )

    def _setup_drop_targets(self):
        """Register drag-and-drop handlers for preview panels and input entry."""
        if not HAS_DND:
            return
        try:
            self.input_entry.drop_target_register('*')
            self.input_entry.dnd_bind('<<Drop>>', self._on_drop_input)
            self._before_canvas.drop_target_register('*')
            self._before_canvas.dnd_bind('<<Drop>>', self._on_drop_before)
            self.root.drop_target_register('*')
            self.root.dnd_bind('<<Drop>>', self._on_drop_input)
        except Exception:
            pass

    # ── Event handlers ────────────────────────────────────────────────

    def _on_drop_input(self, event):
        path = self._strip_braces(event.data)
        if os.path.isfile(path) and self._is_image(path):
            self._load_input(path)

    def _on_drop_before(self, event):
        path = self._strip_braces(event.data)
        if os.path.isfile(path) and self._is_image(path):
            self._load_input(path)

    @staticmethod
    def _strip_braces(path_str):
        """Remove curly braces that Windows may wrap around file paths."""
        s = path_str.strip()
        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]
        return s

    @staticmethod
    def _is_image(path):
        return os.path.splitext(path)[1].lower() in IMAGE_EXTS

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('图片文件', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp'), ('所有文件', '*.*')],
        )
        if path:
            self._load_input(path)

    def _load_input(self, path):
        self.input_path.set(path)
        self.status_var.set(f'已选择: {os.path.basename(path)}')

        # Show preview
        try:
            pil_img = Image.open(path)
            self._input_pil = pil_img
            self._show_in_canvas(self._before_canvas, pil_img)
            # Clear output preview
            self._output_pil = None
            self._after_canvas._pil_image = None
            self._redraw_after_placeholder('处理后将显示\n增强后的图片')
        except Exception as e:
            log(f'无法预览图片: {e}')

    def _browse_output(self):
        path = filedialog.askdirectory(title='选择输出目录')
        if path:
            self.output_dir.set(path)

    def _on_model_change(self, event):
        idx = self.model_combo.current()
        keys = list(MODELS.keys())
        self.model_key.set(keys[idx])
        log(f'切换模型: {keys[idx]}')

    def _open_output_dir(self):
        out_dir = self.output_dir.get()
        if out_dir and os.path.isdir(out_dir):
            self._open_folder(out_dir)
        else:
            messagebox.showinfo('提示', f'输出目录不存在:\n{out_dir}')

    def _save_as(self):
        """Let user save the result image to any location."""
        if not self.result_path or not os.path.isfile(self.result_path):
            messagebox.showwarning('提示', '没有可下载的处理结果。')
            return
        dst = filedialog.asksaveasfilename(
            title='保存增强后的图片',
            defaultextension='.png',
            filetypes=[
                ('PNG 图片', '*.png'),
                ('JPEG 图片', '*.jpg'),
                ('所有文件', '*.*'),
            ],
            initialfile=os.path.basename(self.result_path),
        )
        if dst:
            try:
                import shutil
                shutil.copy2(self.result_path, dst)
                log(f'已保存到: {dst}')
                messagebox.showinfo('提示', f'图片已保存到:\n{dst}')
            except Exception as e:
                messagebox.showerror('错误', f'保存失败:\n{e}')

    @staticmethod
    def _open_folder(path):
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    # ── Preview rendering ─────────────────────────────────────────────

    def _show_in_canvas(self, canvas_frame, pil_img):
        """Resize and display a PIL image in the given canvas frame."""
        canvas = canvas_frame.winfo_children()[0]  # the tk.Canvas inside
        # Resize to fit
        w, h = pil_img.size
        scale = min(self.PREVIEW_MAX_W / w, self.PREVIEW_MAX_H / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        if pil_img.mode == 'RGBA':
            bg = Image.new('RGBA', pil_img.size, (255, 255, 255, 255))
            bg.paste(pil_img, (0, 0), pil_img)
            display = bg.resize((new_w, new_h), Image.LANCZOS)
        elif pil_img.mode != 'RGB':
            display = pil_img.convert('RGB').resize((new_w, new_h), Image.LANCZOS)
        else:
            display = pil_img.resize((new_w, new_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(display)
        canvas._pil_image = photo  # keep ref

        canvas.delete('all')
        canvas.create_image(
            canvas.winfo_width() // 2 if canvas.winfo_width() > 10 else new_w // 2,
            canvas.winfo_height() // 2 if canvas.winfo_height() > 10 else new_h // 2,
            image=photo, anchor=tk.CENTER,
        )

    def _redraw_after_placeholder(self, text):
        canvas = self._after_canvas.winfo_children()[0]
        canvas._pil_image = None
        canvas.delete('all')
        canvas.placeholder = text
        self._draw_placeholder(canvas)

    # ── Processing ────────────────────────────────────────────────────

    def _start_processing(self):
        if self.processing:
            return

        input_path = self.input_path.get().strip()
        if not input_path:
            messagebox.showwarning('提示', '请先选择输入图片。')
            return
        if not os.path.isfile(input_path):
            messagebox.showerror('错误', f'输入文件不存在:\n{input_path}')
            return

        output_dir = self.output_dir.get().strip()
        if not output_dir:
            messagebox.showwarning('提示', '请先选择输出目录。')
            return

        # Disable UI
        self.processing = True
        self.process_btn.configure(state=tk.DISABLED)
        self.download_btn.configure(state=tk.DISABLED)
        self.open_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        self.status_var.set('正在处理中，请稍候...')
        self.result_var.set('（处理中...）')
        self.log_text.delete('1.0', tk.END)

        model_key = list(MODELS.keys())[self.model_combo.current()]
        kwargs = {
            'input_path': input_path,
            'output_dir': output_dir,
            'model_key': model_key,
            'tile': self.tile_size.get(),
            'outscale': self.outscale.get(),
            'fp32': self.fp32.get(),
            'suffix': self.suffix.get() or 'out',
        }
        thread = threading.Thread(target=self._process, kwargs=kwargs, daemon=True)
        thread.start()

    def _process(self, input_path, output_dir, model_key, tile, outscale, fp32, suffix):
        """Background processing thread."""
        result_path = None
        output_img = None
        try:
            log(f'输入图片: {input_path}')
            log(f'输出目录: {output_dir}')
            log(f'模型: {model_key}')
            log(f'参数: tile={tile}, outscale={outscale}, fp32={fp32}')

            # 1. Read image
            log('读取图片...')
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f'无法读取图片: {input_path}')
            log(f'图片尺寸: {img.shape[1]}x{img.shape[0]}, 通道数: {img.shape[2] if len(img.shape) >= 3 else 1}')

            img_mode = 'RGBA' if (len(img.shape) == 3 and img.shape[2] == 4) else None

            # 2. Build upsampler
            log('加载模型...')
            upsampler, model_scale = create_upsampler(
                model_key, tile=tile, tile_pad=10, pre_pad=0,
                fp32=fp32, gpu_id=None,
            )

            # 3. Enhance
            log('正在超分辨率处理中 (这可能需要几分钟)...')
            output, _ = upsampler.enhance(img, outscale=outscale)
            output_img = output
            log(f'处理完成! 输出尺寸: {output.shape[1]}x{output.shape[0]}')

            # 4. Save
            os.makedirs(output_dir, exist_ok=True)
            imgname = os.path.splitext(os.path.basename(input_path))[0]
            ext = os.path.splitext(input_path)[1].lstrip('.')
            if img_mode == 'RGBA':
                ext = 'png'
            save_name = f'{imgname}_{suffix}.{ext}'
            result_path = os.path.join(output_dir, save_name)
            cv2.imwrite(result_path, output)
            log(f'已保存: {result_path}')

        except Exception as e:
            log(f'错误: {e}')
            traceback.print_exc()
            log(traceback.format_exc())

        finally:
            self.root.after(0, self._on_done, result_path, output_img)

    def _on_done(self, result_path, output_img):
        """Called on main thread after processing completes."""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure(state=tk.NORMAL)
        self.result_path = result_path

        if result_path:
            self.status_var.set('处理完成!')
            self.result_var.set(result_path)
            self.download_btn.configure(state=tk.NORMAL)
            self.open_btn.configure(state=tk.NORMAL)

            # Show output preview
            if output_img is not None:
                try:
                    # Convert BGR/RGBA to RGB for PIL display
                    if len(output_img.shape) == 3:
                        if output_img.shape[2] == 4:
                            # RGBA
                            pil_out = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGRA2RGBA))
                        else:
                            pil_out = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
                    else:
                        pil_out = Image.fromarray(output_img)
                    self._output_pil = pil_out
                    self._show_in_canvas(self._after_canvas, pil_out)
                except Exception as e:
                    log(f'预览渲染错误: {e}')

            log(f'\n✅ 完成! 输出文件: {result_path}')
            log(f'   点击 "另存为" 下载到其他位置，或点击 "打开输出目录" 查看。')
        else:
            self.status_var.set('处理失败，请查看日志。')
            self.result_var.set('（处理失败）')
            self.download_btn.configure(state=tk.DISABLED)
            self.open_btn.configure(state=tk.DISABLED)
            log('\n❌ 处理失败，请检查上方日志。')


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    # Try to set app icon
    try:
        icon_path = APP_DIR / 'assets' / 'icon.png'
        if icon_path.exists():
            img = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, img)
    except Exception:
        pass

    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()