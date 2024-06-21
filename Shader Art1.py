# You will need PyOpenGL and pygame installed for this to work.
# Install them via pip: pip install PyOpenGL PyOpenGL_accelerate pygame

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math


# Vertex shader
VERTEX_SHADER = """
#version 330 core
in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment shader
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec2 resolution;
uniform float iTime;

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b*cos(6.28318*(c*t+d));
}

void main()
{
    vec2 uv = gl_FragCoord.xy / resolution - 0.5;
    uv.x *= resolution.x / resolution.y;
    
    float d = length(uv) - 0.5;
    d = abs(sin(d * 8.0 - iTime)) / 8.0;
    
    vec3 color = vec3(0.0);
    color += pow(max(1.0 - abs(d*20.0), 0.0), 2.0) * palette(d, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.1, 0.2));
    
    FragColor = vec4(color, 1.0);
}
"""

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    clock = pygame.time.Clock()

    # Compile shaders and create program
    shader = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
    
    # Define vertex data
    vertices = np.array([
        -1, -1, 0,
         1, -1, 0,
        -1,  1, 0,
         1,  1, 0
    ], dtype=np.float32)

    # Create VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Get position attribute location in shader
    position = glGetAttribLocation(shader, 'position')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Use shader
    glUseProgram(shader)

    iTime = glGetUniformLocation(shader, 'iTime')
    resolution = glGetUniformLocation(shader, 'resolution')

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        glClear(GL_COLOR_BUFFER_BIT)

        # Set uniforms
        glUniform1f(iTime, pygame.time.get_ticks() / 1000.0)
        glUniform2f(resolution, 800.0, 600.0)

        # Draw the quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
