# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, GstBase, Gtk

Gst.init(None)

class GstSink:
    def __init__(self, parent, sink_name, box_name):
        self.parent =  parent
        self.box_name = box_name
        self.gstsample = None
        self.sink_size = None
        self.box = None
        self.glbox = None

        appsink = parent.pipeline.get_by_name(sink_name)
        appsink.connect('new-preroll', self.on_new_sample, True)
        appsink.connect('new-sample', self.on_new_sample, False)

    def transfer_sample(self):
        #print('transfer_sample: {}'.format(self.gstsample))
        gstsample = self.gstsample
        self.gstsample = None
        return gstsample


    def on_new_sample(self, sink, preroll):
        #print('on_new_sample: {}'.format(self))
        if self.gstsample is not None :
            return Gst.FlowReturn.OK

        sample = sink.emit('pull-preroll' if preroll else 'pull-sample')
        #print('on_new_sample B: {}'.format(sample))
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value('width'), s.get_value('height'))
        with self.parent.condition:
            self.gstsample = sample
            self.parent.condition.notify_all()
            #print('on_new_sample E: {}'.format(self.gstsample))
        return Gst.FlowReturn.OK

    def get_box(self):
        if not self.box:
            glbox = self.parent.pipeline.get_by_name(self.box_name)
            print("glbox1: {}".format(glbox))
            if glbox:
                glbox = glbox.get_by_name('filter')
                print("glbox2: {}".format(glbox))
                self.glbox=glbox
            box = self.parent.pipeline.get_by_name('box')
            assert glbox or box
            assert self.sink_size
            if glbox:
                self.box = (glbox.get_property('x'), glbox.get_property('y'),
                        glbox.get_property('width'), glbox.get_property('height'))
            else:
                self.box = (-box.get_property('left'), -box.get_property('top'),
                    self.sink_size[0] + box.get_property('left') + box.get_property('right'),
                    self.sink_size[1] + box.get_property('top') + box.get_property('bottom'))
        return self.box



class GstPipeline:
    def __init__(self, pipeline, user_function, src_size):
        self.user_function = user_function
        self.running = False
        self.src_size = src_size
        self.condition = threading.Condition()

        self.pipeline = Gst.parse_launch(pipeline)
        self.overlay = self.pipeline.get_by_name('overlay')
        self.gloverlay = self.pipeline.get_by_name('gloverlay')
        self.overlaysink = self.pipeline.get_by_name('overlaysink')

        self.sinks = [GstSink(self, 'appsink', 'glbox'), GstSink(self, 'appsink1', 'glbox1')]

        # Set up a pipeline bus watch to catch errors.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        # Set up a full screen window on Coral, no-op otherwise.
        self.setup_window()

    def run(self):
        # Start inference worker.
        self.running = True
        worker = threading.Thread(target=self.inference_loop)
        worker.start()

        # Run pipeline.
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            Gtk.main()
        except:
            pass

        # Clean up.
        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass
        with self.condition:
            self.running = False
            self.condition.notify_all()
        worker.join()

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
            Gtk.main_quit()
        return True

    def inference_loop(self):
        while True:
            with self.condition:
                while not self.sinks[0].gstsample and not self.sinks[1].gstsample and self.running:
                    self.condition.wait()
                if not self.running:
                    break

            for sink_num, sink in enumerate(self.sinks) :
                gstsample = sink.transfer_sample()
                if gstsample is None :
                    continue

                #print("Loop BEGIN {} {}".format(sink_num, sink))
                #print("   sample BEGIN: {}".format(gstsample))
                # Passing Gst.Buffer as input tensor avoids 2 copies of it.
                gstbuffer = gstsample.get_buffer()
                svg, face_coords = self.user_function(gstbuffer, self.src_size, sink.get_box())
                if self.sinks[0] == sink and self.sinks[1].glbox is not None:
                    #print("sink.glbox: {}".format(sink.glbox))
                    print("Face coords: {}".format(face_coords))
                    self.sinks[1].glbox.set_property("crop-x", face_coords[0])
                    self.sinks[1].glbox.set_property("crop-y", face_coords[1])
                    self.sinks[1].glbox.set_property("crop-len", face_coords[2])

                #print("Loop END   {} {}".format(sink_num, sink))
                #print("   sample END: {}".format(sink.gstsample))
                if svg:
                    if self.overlay:
                        self.overlay.set_property('data', svg)
                    if self.gloverlay:
                        self.gloverlay.emit('set-svg', svg, gstbuffer.pts)
                    if self.overlaysink:
                        self.overlaysink.set_property('svg', svg)

    def setup_window(self):
        # Only set up our own window if we have Coral overlay sink in the pipeline.
        if not self.overlaysink:
            return

        gi.require_version('GstGL', '1.0')
        gi.require_version('GstVideo', '1.0')
        from gi.repository import GstGL, GstVideo

        # Needed to commit the wayland sub-surface.
        def on_gl_draw(sink, widget):
            widget.queue_draw()

        # Needed to account for window chrome etc.
        def on_widget_configure(widget, event, overlaysink):
            allocation = widget.get_allocation()
            overlaysink.set_render_rectangle(allocation.x, allocation.y,
                    allocation.width, allocation.height)
            return False

        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.fullscreen()

        drawing_area = Gtk.DrawingArea()
        window.add(drawing_area)
        drawing_area.realize()

        self.overlaysink.connect('drawn', on_gl_draw, drawing_area)

        # Wayland window handle.
        wl_handle = self.overlaysink.get_wayland_window_handle(drawing_area)
        self.overlaysink.set_window_handle(wl_handle)

        # Wayland display context wrapped as a GStreamer context.
        wl_display = self.overlaysink.get_default_wayland_display_context()
        self.overlaysink.set_context(wl_display)

        drawing_area.connect('configure-event', on_widget_configure, self.overlaysink)
        window.connect('delete-event', Gtk.main_quit)
        window.show_all()

        # The appsink pipeline branch must use the same GL display as the screen
        # rendering so they get the same GL context. This isn't automatically handled
        # by GStreamer as we're the ones setting an external display handle.
        def on_bus_message_sync(bus, message, overlaysink):
            if message.type == Gst.MessageType.NEED_CONTEXT:
                _, context_type = message.parse_context_type()
                if context_type == GstGL.GL_DISPLAY_CONTEXT_TYPE:
                    sinkelement = overlaysink.get_by_interface(GstVideo.VideoOverlay)
                    gl_context = sinkelement.get_property('context')
                    if gl_context:
                        display_context = Gst.Context.new(GstGL.GL_DISPLAY_CONTEXT_TYPE, True)
                        GstGL.context_set_gl_display(display_context, gl_context.get_display())
                        message.src.set_context(display_context)
            return Gst.BusSyncReply.PASS

        bus = self.pipeline.get_bus()
        bus.set_sync_handler(on_bus_message_sync, self.overlaysink)

def get_dev_board_model():
  try:
    model = open('/sys/firmware/devicetree/base/model').read().lower()
    if 'mx8mq' in model:
        return 'mx8mq'
    if 'mt8167' in model:
        return 'mt8167'
  except: pass
  return None

def run_pipeline(user_function,
                 src_size,
                 appsink_size,
                 videosrc='/dev/video1',
                 videofmt='raw',
                 headless=False,
                 crop=False,
                 zoom_factor=1.0):

    raw_src_size = src_size

    if videofmt == 'h264':
        SRC_CAPS = 'video/x-h264,width={width},height={height},framerate=30/1'
    elif videofmt == 'jpeg':
        SRC_CAPS = 'image/jpeg,width={width},height={height},framerate=30/1'
    else:
        SRC_CAPS = 'video/x-raw,width={width},height={height},'
        framerate=30
        if src_size[0] == 1280:
            framerate = 10
        elif src_size[0] == 800:
            framerate = 20
        SRC_CAPS += 'framerate={}/1'.format(framerate)

    if videosrc.startswith('/dev/video'):
        PIPELINE = 'v4l2src device=%s ! {src_caps}'%videosrc
    elif videosrc.startswith('http'):
        PIPELINE = 'souphttpsrc location=%s'%videosrc
    elif videosrc.startswith('rtsp'):
        PIPELINE = 'rtspsrc location=%s'%videosrc
    else:
        demux =  'avidemux' if videosrc.endswith('avi') else 'qtdemux'
        PIPELINE = """filesrc location=%s ! %s name=demux  demux.video_0
                    ! queue ! decodebin  ! videorate
                    ! videoconvert n-threads=4 ! videoscale n-threads=4
                    ! {src_caps} ! {leaky_q} """ % (videosrc, demux)

    coral = get_dev_board_model()
    if headless:
        scale = min(appsink_size[0] / src_size[0], appsink_size[1] / src_size[1])
        scale = tuple(int(x * scale) for x in src_size)
        scale_caps = 'video/x-raw,width={width},height={height}'.format(width=scale[0], height=scale[1])
        PIPELINE += """ ! decodebin ! queue ! videoconvert ! videoscale
        ! {scale_caps} ! videobox name=box autocrop=true ! {sink_caps} ! {sink_element}
        """
    elif coral:
        if 'mt8167' in coral:
            PIPELINE += """ ! decodebin ! queue ! v4l2convert ! {scale_caps} !
              glupload ! glcolorconvert ! video/x-raw(memory:GLMemory),format=RGBA !
              tee name=t
                t. ! queue ! glfilterbin filter=glbox name=glbox ! queue ! {sink_caps} ! {sink_element}
                t. ! queue ! glsvgoverlay name=gloverlay sync=false ! glimagesink fullscreen=true
                     qos=false sync=false
            """
            scale_caps = 'video/x-raw,format=BGRA,width={w},height={h}'.format(w=src_size[0], h=src_size[1])
        else:
            if crop:
                # Make video input square to utilize Neural Network input fully (NN inputs are sqare - see inference_box)

                # Method 1 - use custom plugin - glcropbox
                src_size = (src_size[1], src_size[1])
                PIPELINE += """ ! decodebin ! glupload ! tee name=t
                """
                PIPELINE += '  t. ! queue ! glfilterbin filter="glcropbox zoom-factor={zoom_factor}" name=glbox'.format(zoom_factor=zoom_factor)
                PIPELINE += """ ! {sink_caps} ! {sink_element}
                """
                PIPELINE += '  t. ! queue ! glfilterbin filter="glcropbox zoom-factor=1.0" name=glbox1 ! video/x-raw,format=RGB,width=224,height=224 ! appsink name=appsink1 emit-signals=true max-buffers=1 drop=true'
                PIPELINE += '  t. ! queue ! glfilterbin filter="glcropbox zoom-factor={zoom_factor}" name=glbox2'.format(zoom_factor=zoom_factor)
                PIPELINE += ' ! video/x-raw,format=RGB,width={w},height={h} ! glsvgoverlaysink name=overlaysink'.format(w=src_size[0], h=src_size[1])

                # Method 2 - use videocrop plugin
                #additional_crop = 45 # works as zoom factor
                #crop_w = (src_size[0] - src_size[1]) // 2 + additional_crop
                #crop_h = additional_crop
                #src_size = (src_size[1] - additional_crop * 2, src_size[1] - additional_crop * 2)
                #PIPELINE += '! videocrop top={crop_vert} left={crop_horiz} right={crop_horiz} bottom={crop_vert}'.format(crop_horiz=crop_w, crop_vert=crop_h)
                #PIPELINE += """ ! decodebin ! glupload ! tee name=t
                #  t. ! queue ! glfilterbin filter=glcropbox name=glbox ! {sink_caps} ! {sink_element}
                #  t. ! queue ! glfilterbin filter="glcropbox zoom-factor=1.0" name=glbox1 ! video/x-raw,format=RGB,width=224,height=224 ! appsink name=appsink1 emit-signals=true max-buffers=1 drop=true
                #  t. ! queue ! glsvgoverlaysink name=overlaysink
                #"""

            else:
                PIPELINE += """ ! decodebin ! glupload ! tee name=t
                  t. ! queue ! glfilterbin filter=glbox name=glbox ! {sink_caps} ! {sink_element}
                  t. ! queue ! glsvgoverlaysink name=overlaysink
                """
            scale_caps = None
    else:
        scale = min(appsink_size[0] / src_size[0], appsink_size[1] / src_size[1])
        scale = tuple(int(x * scale) for x in src_size)
        scale_caps = 'video/x-raw,width={width},height={height}'.format(width=scale[0], height=scale[1])
        PIPELINE += """ ! tee name=t
            t. ! {leaky_q} ! videoconvert ! videoscale ! {scale_caps} ! videobox name=box autocrop=true
               ! {sink_caps} ! {sink_element}
            t. ! {leaky_q} ! videoconvert
               ! rsvgoverlay name=overlay ! videoconvert ! ximagesink sync=false
            """

    SINK_ELEMENT = 'appsink name=appsink emit-signals=true max-buffers=1 drop=true'
    SINK_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    LEAKY_Q = 'queue max-size-buffers=1 leaky=downstream'

    src_caps = SRC_CAPS.format(width=raw_src_size[0], height=raw_src_size[1])
    sink_caps = SINK_CAPS.format(width=appsink_size[0], height=appsink_size[1])
    pipeline = PIPELINE.format(leaky_q=LEAKY_Q,
        src_caps=src_caps, sink_caps=sink_caps,
        sink_element=SINK_ELEMENT, scale_caps=scale_caps)

    print('Gstreamer pipeline:\n', pipeline)

    pipeline = GstPipeline(pipeline, user_function, src_size)
    pipeline.run()
