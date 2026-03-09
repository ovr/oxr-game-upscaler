# OXR Game Upscaler

> **Experimental project at a very early stage of development.**

Drop-in replacement for FSR3 upscaler/anti-aliasing library that combines multiple upscaling backends under a single runtime-switchable interface.
The main goal of the project is to develop **IMBA** — a neural upscaler capable of competing with modern solutions like DLSS and FSR.

## Upscaler Backends

Switchable at runtime via an in-game imgui overlay (Home key):

| Backend             | Type      | Description                                     |
|---------------------|-----------|-------------------------------------------------|
| **Bilinear**        | Spatial   | Simple bilinear upscale                         |
| **Lanczos**         | Spatial   | Lanczos filter upscale                          |
| **SGSR v1**         | Spatial   | Qualcomm Snapdragon GSR (single-pass)           |
| **SGSRv2 2-Pass**   | Temporal  | SGSRv2 convert + upscale                        |
| **SGSRv2 3-Pass**   | Temporal  | SGSRv2 convert + activate + upscale (default)   |

Optional RCAS sharpening post-pass is available for Bilinear and Lanczos modes.

## Supported Games

- Cyberpunk 2077
