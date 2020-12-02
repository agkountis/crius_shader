mod parsers;

#[derive(Debug, PartialEq)]
pub enum RenderQueue {
    Opaque,
    Transparent(u32),
}

#[derive(Debug, PartialEq)]
pub enum ShaderSource {
    Glsl(String),
    Hlsl(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum OptionalShaderEntryPoints<'a> {
    Geometry(&'a str),
    Tesselation {
        control: &'a str,
        evaluation: &'a str,
    },
}

#[derive(Debug, Default, PartialEq)]
pub struct ShaderEntryPoints<'a> {
    vertex: &'a str,
    fragment: &'a str,
    geometry: Option<&'a str>,
    tesselation: Option<Tesselation<'a>>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Tesselation<'a> {
    control: &'a str,
    evaluation: &'a str,
}

#[derive(Debug, PartialEq)]
pub struct Pass<'a> {
    name: Option<&'a str>,
    shader_entry_points: Option<ShaderEntryPoints<'a>>,
    shader_source: ShaderSource,
}

#[derive(Debug, PartialEq)]
pub struct SubShader<'a> {
    name: Option<&'a str>,
    lod: Option<u32>,
    render_queue: RenderQueue,
    include: Option<String>,
    shader_entry_points: Option<ShaderEntryPoints<'a>>,
    passes: Vec<Pass<'a>>,
}

#[derive(Debug, PartialEq)]
pub struct Shader<'a> {
    name: &'a str,
    include: Option<String>,
    shader_entry_points: Option<ShaderEntryPoints<'a>>,
    sub_shaders: Vec<SubShader<'a>>,
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

#[derive(Debug, PartialEq)]
enum SubShaderMisc<'a> {
    Name(&'a str),
    Include(String),
    Lod(u32),
    EntryPoints(ShaderEntryPoints<'a>),
}

#[derive(Debug, PartialEq)]
enum ShaderMisc {
    Include(String),
}

#[derive(Debug, PartialEq)]
enum PassMisc<'a> {
    Name(&'a str),
    EntryPoints(ShaderEntryPoints<'a>),
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

pub(crate) const ENTRY_POINT_TAG: &str = "entry_points";
pub(crate) const VERTEX_ENTRY_TAG: &str = "vertex";
pub(crate) const FRAGMENT_ENTRY_TAG: &str = "fragment";
pub(crate) const GEOMETRY_ENTRY_TAG: &str = "geometry";
pub(crate) const TESSELATION_BLOCK_TAG: &str = "tesselation";
pub(crate) const TESSELATION_CONTROL_ENTRY_TAG: &str = "control";
pub(crate) const TESSELATION_EVALUATION_ENTRY_TAG: &str = "evaluation";

pub(crate) const GLSL_TAG: &str = "glsl";
pub(crate) const HLSL_TAG: &str = "hlsl";
