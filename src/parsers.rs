use crate::*;

use nom::{
    branch::{alt, permutation},
    bytes::complete::{escaped, is_not, tag, take_until},
    character::complete::{alphanumeric1, digit1, multispace0, one_of},
    combinator::{all_consuming, cut, map, map_res, opt, value, verify},
    error::{context, ContextError, FromExternalError, ParseError, VerboseError},
    multi::{many0, many1, separated_list0},
    number::complete::float,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
use std::num::ParseIntError;

fn ws<'a, F, O, E>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(multispace0, inner, multispace0)
}

fn trim_leading_line_ws<'a, F, E>(parser: F) -> impl FnMut(&'a str) -> IResult<&'a str, String, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, &'a str, E>,
    E: 'a + ParseError<&'a str>,
{
    map(parser, |str: &str| {
        let mut string = str
            .lines()
            .filter(|&line| !line.trim_start().is_empty())
            .fold(String::new(), |acc, line| acc + line.trim_start() + "\n");
        string.truncate(string.len() - 1);
        string.shrink_to_fit();
        string
    })
}

fn parse_esc_str<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    escaped(
        alphanumeric1,
        ESCAPE_CONTROL_CHARACTER,
        one_of(ESCAPABLE_CHARACTERS),
    )(input)
}

pub fn string<'a, E>(i: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    context(
        "string",
        preceded(
            tag(STRING_QUOTE_TAG),
            cut(terminated(
                take_until(STRING_QUOTE_TAG),
                tag(STRING_QUOTE_TAG),
            )),
        ),
    )(i)
}

fn boolean<'a, E>(input: &'a str) -> IResult<&'a str, bool, E>
where
    E: 'a + ParseError<&'a str>,
{
    let parse_true = value(true, tag(TRUE_TAG));
    let parse_false = value(false, tag(FALSE_TAG));
    alt((parse_true, parse_false))(input)
}

fn float2<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let vec2 = preceded(
        ws(tag(BLOCK_OPEN_TAG)),
        cut(terminated(
            verify(
                separated_list0(ws(tag(LIST_SEPARATOR_TAG)), float),
                |list: &[f32]| list.len() == 2,
            ),
            ws(tag(BLOCK_CLOSE_TAG)),
        )),
    );

    context("float2", vec2)(input)
}

fn float3<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let vec3 = preceded(
        ws(tag(BLOCK_OPEN_TAG)),
        cut(terminated(
            verify(
                separated_list0(ws(tag(LIST_SEPARATOR_TAG)), float),
                |list: &[f32]| list.len() == 3,
            ),
            ws(tag(BLOCK_CLOSE_TAG)),
        )),
    );

    context("float3", vec3)(input)
}

fn float4<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let vec4 = preceded(
        ws(tag(BLOCK_OPEN_TAG)),
        cut(terminated(
            verify(
                separated_list0(ws(tag(LIST_SEPARATOR_TAG)), float),
                |list: &[f32]| list.len() == 4,
            ),
            ws(tag(BLOCK_CLOSE_TAG)),
        )),
    );

    context("float4", vec4)(input)
}

fn single_line_comment<'a, E>(i: &'a str) -> IResult<&'a str, (), E>
where
    E: ParseError<&'a str>,
{
    value(
        (), // Output is thrown away.
        pair(tag(SINGLE_LINE_COMMENT_TAG), is_not(NEWLINE_TAG)),
    )(i)
}

fn multi_line_comment<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, (), E> {
    value(
        (), // Output is thrown away.
        tuple((
            tag(MULTI_LINE_COMMENT_OPEN_TAG),
            take_until(MULTI_LINE_COMMENT_CLOSE_TAG),
            tag(MULTI_LINE_COMMENT_CLOSE_TAG),
        )),
    )(i)
}

fn comment<'a, E>(input: &'a str) -> IResult<&'a str, (), E>
where
    E: 'a + ParseError<&'a str>,
{
    let a = ws(single_line_comment);
    let b = ws(multi_line_comment);
    alt((a, b))(input)
}

fn leading_comments<'a, F, O, E>(parser: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: 'a + ParseError<&'a str>,
{
    let maybe_comment = opt(map(many0(ws(comment)), |_| ()));
    preceded(maybe_comment, parser)
}

fn parse_block<'a, F, O, E>(
    open_tag: &'a str,
    inner: F,
    close_tag: &'a str,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    context(
        "parse_block",
        preceded(
            ws(tag(open_tag)),
            cut(terminated(ws(inner), ws(tag(close_tag)))),
        ),
    )
}

fn block<'a, F, O, E>(parser: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    context(
        "block",
        parse_block(BLOCK_OPEN_TAG, parser, BLOCK_CLOSE_TAG),
    )
}

fn string_block<'a, F, O, E>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    context(
        "string_block",
        parse_block(STRING_BLOCK_OPEN_TAG, inner, STRING_BLOCK_CLOSE_TAG),
    )
}

fn named<'a, F, O, E>(name: &'a str, parser: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    context("named", preceded(ws(tag(name)), parser))
}

fn string_block_contents<'a, E>(input: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str>,
{
    take_until(STRING_BLOCK_CLOSE_TAG)(input)
}

fn named_string_block<'a, E>(name: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, String, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    named(
        name,
        string_block(trim_leading_line_ws(string_block_contents)),
    )
}

fn named_block<'a, F, O, E>(
    name: &'a str,
    parser: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    O: 'a,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    named(name, block(parser))
}

fn maybe_named_block<'a, F, O, E>(
    name: &'a str,
    parser: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Option<O>, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    O: 'a,
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    opt(named(name, block(parser)))
}

fn named_value<'a, F, O, E>(
    name: &'a str,
    parser: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    O: 'a,
    E: 'a + ParseError<&'a str>,
{
    separated_value(ws(tag(name)), ws(tag(VALUE_SEPARATOR_TAG)), parser)
}

fn maybe_named_value<'a, F, O, E>(
    name: &'a str,
    parser: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Option<O>, E>
where
    F: 'a + FnMut(&'a str) -> IResult<&'a str, O, E>,
    O: 'a,
    E: 'a + ParseError<&'a str>,
{
    opt(named_value(name, parser))
}

fn entry_point<'a, E>(name: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    named_value(name, string)
}

fn maybe_entry_point<'a, E>(
    name: &'a str,
) -> impl FnMut(&'a str) -> IResult<&'a str, Option<&'a str>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    opt(named_value(name, string))
}

fn optional_entry_points<'a, E>(
    input: &'a str,
) -> IResult<&'a str, Option<OptionalShaderEntryPoints>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let geometry = map(
        entry_point(GEOMETRY_ENTRY_TAG),
        OptionalShaderEntryPoints::Geometry,
    );
    let tesselation = map(
        named_block(
            TESSELATION_BLOCK_TAG,
            pair(
                entry_point(TESSELATION_CONTROL_ENTRY_TAG),
                entry_point(TESSELATION_EVALUATION_ENTRY_TAG),
            ),
        ),
        |(control, evaluation)| OptionalShaderEntryPoints::Tesselation {
            control,
            evaluation,
        },
    );

    opt(alt((geometry, tesselation)))(input)
}

fn entry_points<'a, E>(input: &'a str) -> IResult<&'a str, ShaderEntryPoints, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let vertex = entry_point(VERTEX_ENTRY_TAG);
    let fragment = entry_point(FRAGMENT_ENTRY_TAG);

    let parser = map(
        permutation((
            vertex,
            fragment,
            optional_entry_points,
            optional_entry_points,
        )),
        |(vertex, fragment, opt1, opt2)| {
            let mut entry_points = ShaderEntryPoints {
                vertex,
                fragment,
                ..Default::default()
            };

            for opt_entry_point in vec![opt1, opt2] {
                if let Some(entry_point) = opt_entry_point {
                    match entry_point {
                        OptionalShaderEntryPoints::Geometry(geom) => {
                            entry_points.geometry = Some(geom)
                        }
                        OptionalShaderEntryPoints::Tesselation {
                            control,
                            evaluation,
                        } => {
                            entry_points.tesselation = Some(Tesselation {
                                control,
                                evaluation,
                            })
                        }
                    }
                }
            }

            entry_points
        },
    );

    leading_comments(named_block(ENTRY_POINT_TAG, parser))(input)
}

fn shader_source<'a, E>(input: &'a str) -> IResult<&'a str, ShaderSource, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(alt((
        map(named_string_block(GLSL_TAG), ShaderSource::Glsl),
        map(named_string_block(HLSL_TAG), ShaderSource::Hlsl),
    )))(input)
}

pub fn separated_value<'a, I, O1, O2, O3, E, F, G, H>(
    before: F,
    separator: G,
    value_parser: H,
) -> impl FnMut(I) -> IResult<I, O3, E>
where
    E: 'a + ParseError<I>,
    F: FnMut(I) -> IResult<I, O1, E>,
    G: FnMut(I) -> IResult<I, O2, E>,
    H: FnMut(I) -> IResult<I, O3, E>,
{
    preceded(before, preceded(separator, value_parser))
}

fn name<'a, E>(input: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(separated_value(
        ws(tag(NAME_TAG)),
        ws(tag(VALUE_SEPARATOR_TAG)),
        ws(string),
    ))(input)
}

fn maybe_name<'a, E>(input: &'a str) -> IResult<&'a str, Option<&'a str>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    opt(name)(input)
}

fn lod<'a, E>(input: &'a str) -> IResult<&'a str, u32, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>,
{
    leading_comments(separated_value(
        ws(tag(LOD_TAG)),
        ws(tag(VALUE_SEPARATOR_TAG)),
        map_res(ws(parse_esc_str), |a| a.parse::<u32>()),
    ))(input)
}

fn include<'a, E>(input: &'a str) -> IResult<&'a str, String, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(named_string_block(INCLUDE_TAG))(input)
}

fn pass_misc_value<'a, E>(input: &'a str) -> IResult<&'a str, Option<PassMisc>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    opt(alt((
        map(name, PassMisc::Name),
        map(entry_points, PassMisc::EntryPoints),
    )))(input)
}

fn pass<'a, E>(input: &'a str) -> IResult<&'a str, Pass, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let misc_section = map(
        permutation((pass_misc_value, pass_misc_value)),
        |(opt1, opt2)| {
            let mut maybe_name = None;
            let mut maybe_entry_points = None;

            for opt in vec![opt1, opt2] {
                if let Some(value) = opt {
                    match value {
                        PassMisc::Name(name) => maybe_name = Some(name),
                        PassMisc::EntryPoints(entry_points) => {
                            maybe_entry_points = Some(entry_points)
                        }
                    }
                }
            }

            (maybe_name, maybe_entry_points)
        },
    );

    let parse_pass = leading_comments(named(PASS_TAG, block(pair(misc_section, shader_source))));

    map(
        parse_pass,
        |((name, shader_entry_points), shader_source)| Pass {
            name,
            shader_entry_points,
            shader_source,
        },
    )(input)
}

fn passes<'a, E>(input: &'a str) -> IResult<&'a str, Vec<Pass>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(many1(pass))(input)
}

fn render_queue<'a, E>(input: &'a str) -> IResult<&'a str, RenderQueue, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>,
{
    let parse_val = alt((
        map(tag(RENDER_QUEUE_OPAQUE_TAG), |_| RenderQueue::Opaque),
        map(
            preceded(
                tag(RENDER_QUEUE_TRANSPARENT_TAG),
                delimited(
                    ws(tag(BLOCK_OPEN_TAG)),
                    map_res(digit1, str::parse::<u32>),
                    ws(tag(BLOCK_CLOSE_TAG)),
                ),
            ),
            |num| RenderQueue::Transparent(num),
        ),
    ));

    let parser = preceded(
        ws(tag(RENDER_QUEUE_TAG)),
        preceded(ws(tag(VALUE_SEPARATOR_TAG)), parse_val),
    );

    leading_comments(parser)(input)
}

fn sub_shader_misc_value<'a, E>(input: &'a str) -> IResult<&'a str, Option<SubShaderMisc>, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + ContextError<&'a str>,
{
    opt(alt((
        map(name, SubShaderMisc::Name),
        map(lod, SubShaderMisc::Lod),
        map(include, SubShaderMisc::Include),
        map(entry_points, SubShaderMisc::EntryPoints),
    )))(input)
}

fn sub_shader<'a, E>(input: &'a str) -> IResult<&'a str, SubShader, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + ContextError<&'a str>,
{
    let top_section = map(
        permutation((
            render_queue,
            sub_shader_misc_value,
            sub_shader_misc_value,
            sub_shader_misc_value,
            sub_shader_misc_value,
        )),
        |(render_queue, opt1, opt2, opt3, opt4)| {
            let mut maybe_name = None;
            let mut maybe_lod = None;
            let mut maybe_include = None;
            let mut maybe_entry_points = None;

            for opt in vec![opt1, opt2, opt3, opt4] {
                if let Some(value) = opt {
                    match value {
                        SubShaderMisc::Name(name) => maybe_name = Some(name),
                        SubShaderMisc::Include(include) => maybe_include = Some(include),
                        SubShaderMisc::Lod(lod) => maybe_lod = Some(lod),
                        SubShaderMisc::EntryPoints(entry_points) => {
                            maybe_entry_points = Some(entry_points)
                        }
                    }
                }
            }

            (
                render_queue,
                maybe_name,
                maybe_include,
                maybe_lod,
                maybe_entry_points,
            )
        },
    );

    let contents = map(
        pair(top_section, passes),
        |((render_queue, name, include, lod, shader_entry_points), passes)| SubShader {
            name,
            lod,
            render_queue,
            include,
            shader_entry_points,
            passes,
        },
    );

    named(SUB_SHADER_TAG, block(contents))(input)
}

fn sub_shaders<'a, E>(input: &'a str) -> IResult<&'a str, Vec<SubShader>, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + ContextError<&'a str>,
{
    leading_comments(many1(sub_shader))(input)
}

pub fn shader<'a, E>(input: &'a str) -> IResult<&'a str, Shader, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + ContextError<&'a str>,
{
    let top_section = permutation((name, opt(include), opt(entry_points)));
    let contents = map(
        pair(top_section, sub_shaders),
        |((name, include, shader_entry_points), sub_shaders)| Shader {
            name,
            include,
            shader_entry_points,
            sub_shaders,
        },
    );

    let parser = named(SHADER_TAG, block(contents));

    all_consuming(parser)(input)
}

mod tests {
    use super::*;
    use crate::ShaderSource::Glsl;
    use nom::error::convert_error;

    #[test]
    fn test_single_line_comment_parser() {
        let input = "// My Comment is cool\nHello";
        let (i, _) = single_line_comment::<nom::error::VerboseError<&str>>(input).unwrap();
        assert_eq!(i, "\nHello")
    }

    #[test]
    fn test_multi_line_comment_parser() {
        let input = "/* hello world\n*Multi line*/\nFoo";
        let (i, _) = multi_line_comment::<VerboseError<&str>>(input).unwrap();
        assert_eq!(i, "\nFoo")
    }

    #[test]
    fn test_boolean_parser() {
        let mut input = "true";
        let (_, val) = boolean::<VerboseError<&str>>(input).unwrap();

        assert_eq!(val, true);

        input = "false";
        let (_, val) = boolean::<VerboseError<&str>>(input).unwrap();
        assert_eq!(val, false);
    }

    #[test]
    fn test_string_parser() {
        let inputs = vec![
            r#""MyString""#,
            r#""My String""#,
            r#""My String is! @$#*Awsome""#,
        ];
        let expected = vec![r"MyString", r"My String", r#"My String is! @$#*Awsome"#];

        inputs.iter().zip(expected).for_each(|(&input, expected)| {
            let (_, result) = string::<VerboseError<&str>>(input).unwrap();
            assert_eq!(expected, result);
        });
    }

    #[test]
    fn test_float3() {
        let input = "( 1.0, 2.0, 3.0 )";

        let (_, val) = float3::<VerboseError<&str>>(input).unwrap();

        assert_eq!(val, vec![1.0, 2.0, 3.0])
    }

    #[test]
    fn test_named_string_block() {
        let input = r#"
            glsl ("
                layout(location = 0) in vec3 attrPos;
    
                struct Foo {
                    vec3 bla;
                };
    
                void main()
                {
                    #define FOO #include "bla" int bla = (1 + 1) / 2 * 10;
                }
            ")"#;

        let exp = "layout(location = 0) in vec3 attrPos;
                struct Foo {
                    vec3 bla;
                };
                void main()
                {
                    #define FOO #include \"bla\" int bla = (1 + 1) / 2 * 10;
                }";

        fn f<'a>(tag: &'a str, input: &'a str) -> IResult<&'a str, String, VerboseError<&'a str>> {
            named_string_block::<VerboseError<&str>>(tag)(input)
        };

        match f(GLSL_TAG, input) {
            Err(nom::Err::Failure(e)) | Err(nom::Err::Error(e)) => {
                println!("{}", convert_error(input, e));
                panic!("Parse error!")
            }
            Ok((i, res)) => {
                println!("Result: {}", res);
                println!("Remaining: {}", i);
                assert_eq!(transform_multi_line_str(exp), res)
            }
            _ => {}
        }
    }

    #[test]
    fn test_pass() {
        let input = r#"pass (
                name: "MyPass"
                
                entry_points (
                    vertex: "vert"
                    fragment: "frag"
                    geometry: "geom"
                    tesselation (
                        control: "tess_ctrl"
                        evaluation: "tess_eval"
                    )
                )
                
                glsl("
                    glsl_source
                    glsl_source
                ")
                
            )"#;

        let expected = Pass {
            name: Some("MyPass"),
            shader_entry_points: Some(ShaderEntryPoints {
                vertex: "vert",
                fragment: "frag",
                geometry: Some("geom"),
                tesselation: Some(Tesselation {
                    control: "tess_ctrl",
                    evaluation: "tess_eval",
                }),
            }),
            shader_source: Glsl(String::from("glsl_source\nglsl_source")),
        };

        let (_, res) = pass::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_render_queue() {
        let inputs = vec![
            "render_queue: Opaque",
            "render_queue: Transparent(0)",
            "render_queue: Transparent(10)",
        ];

        let expected = vec![
            RenderQueue::Opaque,
            RenderQueue::Transparent(0),
            RenderQueue::Transparent(10),
        ];

        inputs.iter().zip(expected).for_each(|(&input, expected)| {
            let (_, res) = render_queue::<VerboseError<&str>>(input).unwrap();
            assert_eq!(res, expected)
        })
    }

    #[test]
    fn test_optional_name() {
        let input = r#"name:"My Foo""#;
        let expected = Some("My Foo");

        let (_, res) = maybe_name::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_sub_shader() {
        let input = r#"
        sub_shader (
             render_queue: Transparent(1)
             lod: 600
             name: "Sub Shader 1"
    
            include ("
                #define MY_DEFINE
            ")
    
            // bla
            /*
             * foo
             * bla
             */
             entry_points (
                vertex: "vert"
                fragment: "frag"
             )
    
            pass ( /*
                * foo
                */
                    name: "Pass1" //foo
    
                    glsl("
                        glsl_code
                        glsl_code
                    ")
            )
    
            pass (
                hlsl("
                    hlsl_code
                    hlsl_code
                ")
            )
        )
        "#;

        let expected = SubShader {
            name: Some("Sub Shader 1"),
            lod: Some(600),
            render_queue: RenderQueue::Transparent(1),
            include: Some("#define MY_DEFINE".to_string()),
            shader_entry_points: Some(ShaderEntryPoints {
                vertex: "vert",
                fragment: "frag",
                geometry: None,
                tesselation: None,
            }),
            passes: vec![
                Pass {
                    name: Some("Pass1"),
                    shader_entry_points: None,
                    shader_source: ShaderSource::Glsl(String::from("glsl_code\nglsl_code")),
                },
                Pass {
                    name: None,
                    shader_entry_points: None,
                    shader_source: ShaderSource::Hlsl(String::from("hlsl_code\nhlsl_code")),
                },
            ],
        };

        let (_, res) = sub_shader::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_shader() {
        let input = r#"
        shader (
            name: "Test Shader"
    
            include ("
                #include "this.glsl"
            ")
            
            sub_shader(
                name: "SubShader!1"
                lod: 700
    
                render_queue: Opaque
                
                entry_points (
                    vertex: "vert"
                    fragment: "frag"
                )
    
                pass (
                    name: "Best pass"
                    glsl("
                        glsl_code
                        glsl_code
                    ")
                )
            )
        )
        "#;

        let exp = Shader {
            name: "Test Shader",
            include: Some(r#"#include "this.glsl""#.to_string()),
            shader_entry_points: None,
            sub_shaders: vec![SubShader {
                name: Some("SubShader!1"),
                lod: Some(700),
                render_queue: RenderQueue::Opaque,
                include: None,
                shader_entry_points: Some(ShaderEntryPoints {
                    vertex: "vert",
                    fragment: "frag",
                    geometry: None,
                    tesselation: None,
                }),
                passes: vec![Pass {
                    name: Some("Best pass"),
                    shader_entry_points: None,
                    shader_source: ShaderSource::Glsl(String::from("glsl_code\nglsl_code")),
                }],
            }],
        };

        let (i, res) = shader::<VerboseError<&str>>(input).unwrap();
        assert_eq!(exp, res);
        assert_eq!(i, "");
    }

    fn transform_multi_line_str<T: AsRef<str>>(input: T) -> String {
        let mut output = input
            .as_ref()
            .lines()
            .filter(|line| !line.trim_start().is_empty())
            .fold(String::new(), |acc, line| acc + line.trim_start() + "\n");
        output.truncate(output.len() - 1);
        output.shrink_to_fit();
        output
    }
}
