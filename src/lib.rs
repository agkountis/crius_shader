mod parsers;

#[derive(Debug, PartialEq, Clone)]
pub enum RenderQueue {
    Opaque,
    Transparent(u32),
}

#[derive(Debug, PartialEq, Clone)]
pub enum OptionalShaderSource {
    Geometry(String),
    Tesselation { control: String, evaluation: String },
}

#[derive(Debug, Default, PartialEq)]
pub struct Tesselation {
    control: String,
    evaluation: String,
}

#[derive(Debug, Default, PartialEq)]
pub struct ShadersSources {
    vertex: String,
    fragment: String,
    geometry: Option<String>,
    tesselation: Option<Tesselation>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Pass {
    name: Option<String>,
    shaders: ShadersSources,
}

#[derive(Debug, PartialEq)]
pub struct SubShader {
    name: Option<String>,
    lod: Option<u32>,
    render_queue: RenderQueue,
    include: Option<String>,
    passes: Vec<Pass>,
}

#[derive(Debug, PartialEq)]
pub enum FormatValue {
    Str(String),
    Boolean(bool),
    Float(f32),
    Vector2([f32; 2]),
    Vector3([f32; 3]),
    Vector4([f32; 4]),
}

pub(crate) const SHADING_LANGUAGE_TAG: &str = "shading_language";
pub(crate) const NAME_TAG: &str = "name";
pub(crate) const SHADER_TAG: &str = "shader";
pub(crate) const LOD_TAG: &str = "lod";
pub(crate) const SHADER_FEATURES_TAG: &str = "shader_features";
pub(crate) const RENDER_QUEUE_TAG: &str = "render_queue";
pub(crate) const PASS_TAG: &str = "pass";

pub(crate) const VERTEX_SHADER_TAG: &str = "vertex";
pub(crate) const FRAGMENT_SHADER_TAG: &str = "fragment";
pub(crate) const GEOMETRY_SHADER_TAG: &str = "geometry";
pub(crate) const TESSELATION_BLOCK_TAG: &str = "tesselation";
pub(crate) const TESSELATION_CONTROL_SHADER_TAG: &str = "control";
pub(crate) const TESSELATION_EVALUATION_SHADER_TAG: &str = "evaluation";
