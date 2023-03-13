# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities."""
import collections
import io
import time

SVG_HEADER = '<svg width="{w}" height="{h}" version="1.1" >'
SVG_RECT = '<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="{s}" stroke-width="{sw}" stroke-opacity="{so}" fill="{fill}" />'
SVG_SOLID_RECT = '<rect x="{x}" y="{y}" width="{w}" height="{h}"  fill-opacity="{so}" fill="{fill}" />'
SVG_TEXT = '''
<text x="{x}" y="{y}" font-size="{fs}" dx="0.05em" dy="0.05em" fill="black">{t}</text>
<text x="{x}" y="{y}" font-size="{fs}" fill="white">{t}</text>
'''
SVG_POINTER = '''
<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{arc_color}" stroke-width="{arc_sw}" fill="none" />
<circle cx="{cx}" cy="{cy}" r="1" stroke="{line_color}" stroke-width="{arc_sw}" />
<line x1="{cx}" y1="{y1a}" x2="{cx}" y2="{y2a}" stroke="{line_color}" stroke-width="{line_sw}" />
<line x1="{cx}" y1="{y1b}" x2="{cx}" y2="{y2b}" stroke="{line_color}" stroke-width="{line_sw}" />
<line x1="{x1a}" y1="{cy}" x2="{x2a}" y2="{cy}" stroke="{line_color}" stroke-width="{line_sw}" />
<line x1="{x1b}" y1="{cy}" x2="{x2b}" y2="{cy}" stroke="{line_color}" stroke-width="{line_sw}" />
'''
SVG_FOOTER = '</svg>'

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

class SVG:
    def __init__(self, size):
        self.io = io.StringIO()
        self.io.write(SVG_HEADER.format(w=size[0] , h=size[1]))

    def add_rect(self, x, y, w, h, stroke, stroke_width, stroke_opacity, fill="none"):
        self.io.write(SVG_RECT.format(x=x, y=y, w=w, h=h, s=stroke, sw=stroke_width, so=stroke_opacity, fill=fill))

    def add_solid_rect(self, x, y, w, h, fill_opacity, fill="none"):
        self.io.write(SVG_SOLID_RECT.format(x=x, y=y, w=w, h=h, so=fill_opacity, fill=fill))

    def add_text(self, x, y, text, font_size):
        self.io.write(SVG_TEXT.format(x=x, y=y, t=text, fs=font_size))

    def add_pointer(self, x, y, d):
        d2 = d / 2.0
        d4 = d / 4.0
        r = d2 * 0.75
        cx = x + d2
        cy = y + d2
        y1a = y
        y2a = y + d4
        y1b = y + d
        y2b = y + d - d4
        x1a = x
        x2a = x + d4
        x1b = x + d
        x2b = x + d - d4
        self.io.write(SVG_POINTER.format(cx=cx, cy=cy, r=r, y1a=y1a, y2a=y2a, y1b=y1b, y2b=y2b, x1a=x1a, x2a=x2a, x1b=x1b, x2b=x2b, arc_color="red", arc_sw=3, line_color="white", line_sw=2))

    def finish(self):
        self.io.write(SVG_FOOTER)
        return self.io.getvalue()
