# crius_shader

![CI](https://github.com/crius-engine/crius_shader/workflows/CI/badge.svg)

Parser for crius engine's shader file format.

The shader file format is heavily inspired by Unity's ShaderLab and Filament's material definition files.

Example shader (early stages):
```
shader (
    name: "My Shader"

    include("
        // Include block to be included in every pass of every sub-shader.
    ")

    // Sub shaders to support different shader LOD levels
    sub_shader (
        name: "Sub shader 1"
        lod: 600

        // Similar to Shaderlab's shader keywords for multi_compile directives
        shader_keywords (
            (NORMAL_MAPPING_ON, NORMAL_MAPPING_OFF)
            (_, SHADER_FEATURE_PARALLAX_OCCLUSION_MAPPING)
        )

        // The render queue this shader should be rendered in.
        // can be Opaque or Transparent(n) where n is a number in the range [0, n] to determine order 
        // for transparent rendering.
        render_queue: Opaque

        include ("
            // Include block to be included to each pass of this sub-shader.
        ")

        // Represents a render pass. At least 1 must be defined.
        pass (
            name: "My Pass"

            // Shader program entry points. 
            // Defining entry points in the pass overrides any defined in the sub-shader level.
            entry_points (
                vertex: "vert"
                fragment: "frag"
                geometry: "geom"
                tesselation (
                    control: "tess_ctrl"
                    evaluation: "tess_eval"
                )
            )

            glsl ("
                void vert() 
                {
                    // Vertex program
                }
            
                void frag()
                {
                    // Fragment program
                }

                void geom()
                {
                    // Geometry program
                }

                void tess_ctrl()
                {
                    // Tesselation control program
                }

                void tess_eval()
                {
                    // Tesselation evaluation program
                }

            ")
        )
    )
)
```
