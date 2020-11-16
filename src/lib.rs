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

#[derive(Debug, PartialEq, Clone)]
pub enum SubShaderMisc {
    Name(String),
    Include(String),
    Lod(u32),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ShaderMisc {
    Include(String),
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
pub struct Shader {
    name: String,
    include: Option<String>,
    sub_shaders: Vec<SubShader>,
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

pub(crate) const NAME_TAG: &str = "name";
pub(crate) const SHADER_TAG: &str = "shader";
pub(crate) const SUB_SHADER_TAG: &str = "sub_shader";
pub(crate) const LOD_TAG: &str = "lod";
pub(crate) const SHADER_FEATURES_TAG: &str = "shader_features";
pub(crate) const RENDER_QUEUE_TAG: &str = "render_queue";
pub(crate) const RENDER_QUEUE_OPAQUE_TAG: &str = "Opaque";
pub(crate) const RENDER_QUEUE_TRANSPARENT_TAG: &str = "Transparent";
pub(crate) const PASS_TAG: &str = "pass";
pub(crate) const PASSES_TAG: &str = "passes";
pub(crate) const INCLUDE_TAG: &str = "include";

pub(crate) const VALUE_SEPARATOR_TAG: &str = ":";
pub(crate) const LIST_SEPARATOR_TAG: &str = ",";
pub(crate) const BLOCK_OPEN_TAG: &str = "(";
pub(crate) const BLOCK_CLOSE_TAG: &str = ")";
pub(crate) const STRING_BLOCK_OPEN_TAG: &str = r#"(""#;
pub(crate) const STRING_BLOCK_CLOSE_TAG: &str = r#"")"#;
pub(crate) const STRING_QUOTE_TAG: &str = r#"""#;

pub(crate) const ESCAPABLE_CHARACTERS: &str = r#""n\"#;
pub(crate) const ESCAPE_CONTROL_CHARACTER: char = '\\';

pub(crate) const TRUE_TAG: &str = "true";
pub(crate) const FALSE_TAG: &str = "false";

pub(crate) const SINGLE_LINE_COMMENT_TAG: &str = "//";
pub(crate) const MULTI_LINE_COMMENT_OPEN_TAG: &str = "/*";
pub(crate) const MULTI_LINE_COMMENT_CLOSE_TAG: &str = "*/";

pub(crate) const NEWLINE_TAG: &str = "\n\r";

pub(crate) const VERTEX_SHADER_TAG: &str = "vertex";
pub(crate) const FRAGMENT_SHADER_TAG: &str = "fragment";
pub(crate) const GEOMETRY_SHADER_TAG: &str = "geometry";
pub(crate) const TESSELATION_BLOCK_TAG: &str = "tesselation";
pub(crate) const TESSELATION_CONTROL_SHADER_TAG: &str = "control";
pub(crate) const TESSELATION_EVALUATION_SHADER_TAG: &str = "evaluation";
