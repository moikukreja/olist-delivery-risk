/**
 * ShaderBackground.tsx
 * --------------------
 * A slowly drifting aurora, drawn by the graphics card using raw WebGL.
 *
 * HOW THIS WORKS, IN PLAIN TERMS
 * ------------------------------
 * Normally the CPU decides the colour of things. Here we hand a tiny program
 * called a SHADER to the graphics card (GPU) instead. The GPU then runs that
 * program once for EVERY PIXEL on screen, all at the same time - which is why
 * it can repaint a full-screen animation 60 times a second without the page
 * stuttering.
 *
 * There are two shaders:
 *   VERTEX shader   - decides WHERE things are. Ours is trivial: it just
 *                     stretches two triangles to cover the whole screen.
 *   FRAGMENT shader - decides WHAT COLOUR each pixel is. All the interesting
 *                     work happens here.
 *
 * The colour comes from "fractal noise": we add several layers of smooth
 * random noise at different sizes, the same trick used to generate clouds and
 * terrain in games. Feeding time into it makes the whole thing drift.
 *
 * If the browser has no WebGL support, the component quietly renders nothing
 * and the plain dark background shows through instead.
 */

import { useEffect, useRef } from "react";

const VERTEX_SHADER = `
attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision highp float;

uniform vec2  u_resolution;
uniform float u_time;

// --- smooth value noise -------------------------------------------------
// A repeatable "random" number for any 2D point. Same input always gives the
// same output, which is what stops the animation flickering.
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Random values at whole-number grid points, smoothly blended in between.
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);   // smoothstep curve - removes hard edges
  return mix(
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

// Stack five layers of noise: each one twice as fine and half as strong as
// the last. This is what turns bland blobs into something organic.
float fbm(vec2 p) {
  float total = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < 5; i++) {
    total += amplitude * noise(p);
    p *= 2.02;
    amplitude *= 0.5;
  }
  return total;
}

void main() {
  // Normalise pixel coordinates and correct for the window's aspect ratio,
  // so the pattern is not stretched on a wide monitor.
  vec2 uv = gl_FragCoord.xy / u_resolution.xy;
  vec2 p = uv;
  p.x *= u_resolution.x / u_resolution.y;

  float t = u_time * 0.028;

  // Domain warping: use noise to distort the input of more noise. This is the
  // single trick that makes the result look like flowing light rather than
  // static fog.
  vec2 q = vec2(fbm(p * 2.4 + vec2(0.0, t)),
                fbm(p * 2.4 + vec2(4.7, -t)));
  vec2 r = vec2(fbm(p * 2.4 + 3.6 * q + vec2(1.7, 9.2) + 0.20 * t),
                fbm(p * 2.4 + 3.6 * q + vec2(8.3, 2.8) + 0.14 * t));
  float f = fbm(p * 2.4 + 3.6 * r);

  // Map the noise value onto the app's palette: deep navy -> indigo ->
  // electric blue -> a hint of cyan on the brightest ridges.
  vec3 deepNavy  = vec3(0.024, 0.039, 0.098);
  vec3 indigo    = vec3(0.086, 0.106, 0.290);
  vec3 blue      = vec3(0.180, 0.310, 0.720);
  vec3 cyan      = vec3(0.430, 0.900, 1.000);

  vec3 colour = mix(deepNavy, indigo, clamp(f * 1.7, 0.0, 1.0));
  colour = mix(colour, blue, clamp(pow(r.x, 1.8) * 0.85, 0.0, 1.0));
  colour = mix(colour, cyan, clamp(pow(q.y, 3.2) * 0.30, 0.0, 1.0));

  // Vignette: darken the corners so the glass panels in the middle pop.
  float vignette = smoothstep(1.30, 0.28, length(uv - 0.5));
  colour *= 0.42 + 0.58 * vignette;

  // A very faint grain breaks up colour banding on large flat gradients.
  colour += (hash(gl_FragCoord.xy) - 0.5) * 0.014;

  gl_FragColor = vec4(colour, 1.0);
}
`;

function compile(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.warn("Shader failed to compile:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export default function ShaderBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      powerPreference: "low-power",
    });
    // Older machines and locked-down browsers may have no WebGL at all. That
    // is fine - we simply leave the canvas blank and the CSS background shows.
    if (!gl) return;

    const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    if (!vertex || !fragment) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn("Shader program failed to link:", gl.getProgramInfoLog(program));
      return;
    }
    gl.useProgram(program);

    // Two triangles that together cover the entire screen. The fragment
    // shader then colours every pixel inside them.
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    const positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const timeLocation = gl.getUniformLocation(program, "u_time");

    // Render at half resolution. The image is a soft blur anyway, so nobody
    // can tell - but it roughly quarters the GPU work, which matters on the
    // free Hugging Face hardware and on laptops running from battery.
    const SCALE = 0.5;

    function resize() {
      if (!canvas || !gl) return;
      const width = Math.floor(window.innerWidth * SCALE);
      const height = Math.floor(window.innerHeight * SCALE);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
      gl.uniform2f(resolutionLocation, width, height);
    }

    resize();
    window.addEventListener("resize", resize);

    // Honour the operating-system "reduce motion" setting: draw one still
    // frame instead of animating.
    const stillOnly = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let frame = 0;
    const start = performance.now();

    function draw() {
      if (!gl) return;
      gl.uniform1f(timeLocation, (performance.now() - start) / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (!stillOnly) frame = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      gl.deleteProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
      gl.deleteBuffer(buffer);
    };
  }, []);

  return <canvas ref={canvasRef} className="shader-bg" aria-hidden="true" />;
}
