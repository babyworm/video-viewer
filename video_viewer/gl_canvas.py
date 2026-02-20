from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLTexture, QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLVersionProfile, QOpenGLVersionFunctionsFactory
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QSurfaceFormat, QPainter, QColor
import numpy as np
import ctypes

class GLCanvas(QOpenGLWidget):
    mouse_moved = pyqtSignal(int, int)

    # Vertex Shader: Pass-through positions and texture coordinates
    VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
    """

    # Fragment Shader: YUV to RGB conversion
    FRAGMENT_SHADER_I420 = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;

    uniform sampler2D texY;
    uniform sampler2D texU;
    uniform sampler2D texV;

    void main() {
        float y = texture(texY, TexCoord).r;
        float u = texture(texU, TexCoord).r - 0.5;
        float v = texture(texV, TexCoord).r - 0.5;

        float r = y + 1.402 * v;
        float g = y - 0.344136 * u - 0.714136 * v;
        float b = y + 1.772 * u;

        FragColor = vec4(r, g, b, 1.0);
    }
    """

    # RGB Shader (Pass-through for when we have RGB data)
    FRAGMENT_SHADER_RGB = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;

    uniform sampler2D texRGB;

    void main() {
        FragColor = texture(texRGB, TexCoord);
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.textureY = None
        self.textureU = None
        self.textureV = None
        self.textureRGB = None

        self.shaderProgramI420 = None
        self.shaderProgramRGB = None

        self.vao = None
        self.vbo = None

        self.current_mode = "RGB" # or "I420"
        self.image_width = 0
        self.image_height = 0

        self.grid_size = 0
        self.gl = None # OpenGL Functions Object

        # Pending data for upload if context not ready
        self.pending_i420 = None
        self.pending_rgb = None

        # Pending data for upload if context not ready
        self.pending_i420 = None
        self.pending_rgb = None

    def initializeGL(self):
        # Request OpenGL 3.3 Core or higher (using 4.1 here as requested previously, but 3.3 is safer fallback)
        # Let's try 3.3 Core as it's more widely supported than 4.1 on some linux drivers,
        # providing enough features for our shaders.
        # However, previous error was about casting to 4.1.
        # Let's stick to a profile.

        profile = QOpenGLVersionProfile()
        profile.setVersion(4, 1)
        profile.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

        self.gl = QOpenGLVersionFunctionsFactory.get(profile, self.context())
        if not self.gl:
             # Fallback or error
             print("Failed to get OpenGL 3.3 Core functions. Trying default functions.")
             self.gl = self.context().functions()

        self.gl.initializeOpenGLFunctions()
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        # Compile Shaders
        self.shaderProgramI420 = self.create_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER_I420)
        self.shaderProgramRGB = self.create_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER_RGB)

        # Setup VAO/VBO
        vertices = np.array([
            # Positions   # TexCoords
             1.0,  1.0, 0.0,  1.0, 0.0,
             1.0, -1.0, 0.0,  1.0, 1.0,
            -1.0, -1.0, 0.0,  0.0, 1.0,
            -1.0,  1.0, 0.0,  0.0, 0.0
        ], dtype=np.float32)

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        self.vbo = QOpenGLBuffer()
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)

        # Position
        self.shaderProgramRGB.bind()
        self.gl.glEnableVertexAttribArray(0)
        self.gl.glVertexAttribPointer(0, 3, 0x1406, False, 5 * 4, ctypes.c_void_p(0))

        # TexCoord
        self.gl.glEnableVertexAttribArray(1)
        self.gl.glVertexAttribPointer(1, 2, 0x1406, False, 5 * 4, ctypes.c_void_p(3 * 4))

        self.vao.release()
        self.vbo.release()

        # Check pending uploads
        if self.pending_i420:
             self.set_image_i420(*self.pending_i420)
             self.pending_i420 = None
        elif self.pending_rgb:
             self.set_image_rgb(*self.pending_rgb)
             self.pending_rgb = None

    def create_program(self, vert_src, frag_src):
        program = QOpenGLShaderProgram()
        program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vert_src)
        program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag_src)
        program.link()
        return program

    def resizeGL(self, w, h):
        if self.gl:
            self.gl.glViewport(0, 0, w, h)

    def paintGL(self):
        if not self.gl: return
        # print("DEBUG: paintGL drawing") # Too spammy for frame loop, maybe once?

        self.gl.glClear(0x00004000) # GL_COLOR_BUFFER_BIT

        if self.current_mode == "I420":
            if self.textureY and self.textureU and self.textureV:
                self.shaderProgramI420.bind()
                self.vao.bind()

                # Bind Textures
                self.gl.glActiveTexture(0x84C0) # GL_TEXTURE0
                self.textureY.bind()
                self.shaderProgramI420.setUniformValue("texY", 0)

                self.gl.glActiveTexture(0x84C1) # GL_TEXTURE1
                self.textureU.bind()
                self.shaderProgramI420.setUniformValue("texU", 1)

                self.gl.glActiveTexture(0x84C2) # GL_TEXTURE2
                self.textureV.bind()
                self.shaderProgramI420.setUniformValue("texV", 2)

                self.gl.glDrawArrays(0x0006, 0, 4) # GL_TRIANGLE_FAN

                self.textureY.release()
                self.textureU.release()
                self.textureV.release()
                self.vao.release()
                self.shaderProgramI420.release()

        elif self.current_mode == "RGB":
            if self.textureRGB:
                self.shaderProgramRGB.bind()
                self.vao.bind()

                self.gl.glActiveTexture(0x84C0)
                self.textureRGB.bind()
                self.shaderProgramRGB.setUniformValue("texRGB", 0)

                self.gl.glDrawArrays(0x0006, 0, 4)

                self.textureRGB.release()
                self.vao.release()
                self.shaderProgramRGB.release()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.grid_size > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QColor(0, 255, 0))
            # Overlay grid logic can be added here
            painter.end()


    def set_image_i420(self, y_data, u_data, v_data, width, height, stride_y, stride_uv):
        if not self.isValid():
            self.pending_i420 = (y_data, u_data, v_data, width, height, stride_y, stride_uv)
            self.pending_rgb = None
            self.current_mode = "I420"
            return

        self.makeCurrent() # Ensure context is current
        self.current_mode = "I420"
        self.image_width = width
        self.image_height = height

        # Y Plane
        if self.textureY is None:
            self.textureY = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            self.textureY.create()

        self.textureY.setSize(stride_y, height)
        self.textureY.setFormat(QOpenGLTexture.TextureFormat.R8_UNorm)
        self.textureY.allocateStorage(QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8)
        self.textureY.setData(0, QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8, y_data)
        self.textureY.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        self.textureY.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)

        # U Plane
        if self.textureU is None:
            self.textureU = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            self.textureU.create()

        self.textureU.setSize(stride_uv, height // 2)
        self.textureU.setFormat(QOpenGLTexture.TextureFormat.R8_UNorm)
        self.textureU.allocateStorage(QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8)
        self.textureU.setData(0, QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8, u_data)
        self.textureU.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        self.textureU.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)

        # V Plane
        if self.textureV is None:
            self.textureV = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            self.textureV.create()

        self.textureV.setSize(stride_uv, height // 2)
        self.textureV.setFormat(QOpenGLTexture.TextureFormat.R8_UNorm)
        self.textureV.allocateStorage(QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8)
        self.textureV.setData(0, QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.UInt8, v_data)
        self.textureV.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        self.textureV.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)

        self.doneCurrent()
        self.update()

    def set_image_rgb(self, rgb_data, width, height, stride):
        if not self.isValid():
             self.pending_rgb = (rgb_data, width, height, stride)
             self.pending_i420 = None
             self.current_mode = "RGB"
             return

        self.makeCurrent()
        self.current_mode = "RGB"
        self.image_width = width
        self.image_height = height

        if self.textureRGB is None:
            self.textureRGB = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            self.textureRGB.create()

        self.textureRGB.setSize(width, height)
        self.textureRGB.setFormat(QOpenGLTexture.TextureFormat.RGB8_UNorm)
        self.textureRGB.allocateStorage(QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.UInt8)
        if hasattr(rgb_data, 'tobytes'):
            self.textureRGB.setData(0, QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.UInt8, rgb_data.tobytes())
        else:
            self.textureRGB.setData(0, QOpenGLTexture.PixelFormat.RGB, QOpenGLTexture.PixelType.UInt8, rgb_data)
        self.textureRGB.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        self.textureRGB.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)

        self.doneCurrent()
        self.update()

    def set_grid(self, size):
        self.grid_size = size
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.image_width > 0 and self.image_height > 0:
             x = int(event.position().x() / self.width() * self.image_width)
             y = int(event.position().y() / self.height() * self.image_height)
             self.mouse_moved.emit(x, y)
