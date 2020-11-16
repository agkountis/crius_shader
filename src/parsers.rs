use crate::*;

use nom::multi::many0;
use nom::{
    branch::{alt, permutation},
    bytes::complete::{escaped, is_not, tag, take_until},
    character::complete::{alphanumeric1, digit1, multispace0, one_of},
    combinator::{cut, map, map_res, opt, value, verify},
    error::{context, ContextError, FromExternalError, ParseError, VerboseError},
    multi::{many1, separated_list0},
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

fn trim_line_leading_ws<'a, F, E>(parser: F) -> impl FnMut(&'a str) -> IResult<&'a str, String, E>
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
        string_block(trim_line_leading_ws(string_block_contents)),
    )
}

fn shader_source<'a, E>(shader_tag: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, String, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(named_string_block(shader_tag))
}

fn optional_shader_source<'a, E>(
    input: &'a str,
) -> IResult<&'a str, Option<OptionalShaderSource>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let tesselation_ctrl = shader_source(TESSELATION_CONTROL_SHADER_TAG);
    let tesselation_eval = shader_source(TESSELATION_EVALUATION_SHADER_TAG);
    let geometry = shader_source(GEOMETRY_SHADER_TAG);
    let tesselation_block = leading_comments(named(
        TESSELATION_BLOCK_TAG,
        block(permutation((tesselation_ctrl, tesselation_eval))),
    ));

    let tesselation = map(tesselation_block, |(control, evaluation)| {
        OptionalShaderSource::Tesselation {
            control,
            evaluation,
        }
    });

    opt(alt((
        map(geometry, OptionalShaderSource::Geometry),
        tesselation,
    )))(input)
}

fn shader_sources<'a, E>(input: &'a str) -> IResult<&'a str, ShadersSources, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let mut parser = map(
        permutation((
            shader_source(VERTEX_SHADER_TAG),
            shader_source(FRAGMENT_SHADER_TAG),
            optional_shader_source,
            optional_shader_source,
            optional_shader_source,
        )),
        |(vertex, fragment, opt1, opt2, opt3)| {
            let mut shaders = ShadersSources {
                vertex,
                fragment,
                ..Default::default()
            };

            for opt_shader in vec![opt1, opt2, opt3].into_iter() {
                if let Some(source) = opt_shader {
                    match source {
                        OptionalShaderSource::Geometry(s) => shaders.geometry = Some(s),
                        OptionalShaderSource::Tesselation {
                            control,
                            evaluation,
                        } => {
                            shaders.tesselation = Some(Tesselation {
                                control,
                                evaluation,
                            })
                        }
                    }
                }
            }

            shaders
        },
    );

    parser(input)
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

pub fn maybe_name<'a, E>(input: &'a str) -> IResult<&'a str, Option<String>, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    leading_comments(opt(map(
        separated_value(ws(tag(NAME_TAG)), ws(tag(VALUE_SEPARATOR_TAG)), ws(string)),
        String::from,
    )))(input)
}

pub fn maybe_lod<'a, E>(input: &'a str) -> IResult<&'a str, Option<u32>, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>,
{
    leading_comments(opt(separated_value(
        ws(tag(LOD_TAG)),
        ws(tag(VALUE_SEPARATOR_TAG)),
        map_res(ws(parse_esc_str), |a| a.parse::<u32>()),
    )))(input)
}

pub fn pass<'a, E>(input: &'a str) -> IResult<&'a str, Pass, E>
where
    E: 'a + ParseError<&'a str> + ContextError<&'a str>,
{
    let parse_pass = leading_comments(named(PASS_TAG, block(tuple((maybe_name, shader_sources)))));
    map(parse_pass, |(name, shaders)| Pass { name, shaders })(input)
}

pub fn render_queue<'a, E>(input: &'a str) -> IResult<&'a str, RenderQueue, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>,
{
    let parse_val = alt((
        value(RenderQueue::Opaque, ws(tag(RENDER_QUEUE_OPAQUE_TAG))),
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

pub fn sub_shader<'a, E>(input: &'a str) -> IResult<&'a str, SubShader, E>
where
    E: 'a + ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + ContextError<&'a str>,
{
    let maybe_include = leading_comments(opt(named_string_block("include")));
    let passes = leading_comments(named(PASSES_TAG, block(many1(pass))));

    let contents = map(
        tuple((maybe_name, maybe_lod, render_queue, maybe_include, passes)),
        |(name, lod, render_queue, include, passes)| SubShader {
            name,
            lod,
            render_queue,
            include: include.map(String::from),
            passes,
        },
    );

    named(SUB_SHADER_TAG, block(contents))(input)
}

mod tests {
    use super::*;
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
            vertex ("
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

        match f(VERTEX_SHADER_TAG, input) {
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
    fn test_shaders() {
        let input = r#"
            // Test comment
            /**
            * Test comment
            */
            
            // Test comment
            vertex ("
                //Test comment
                /*Test comment
                * hello
                */
                /* Test comment */
                #define FOO #include "bla" int bla = (1 + 1) / 2 * 10;
            ")
            
            fragment ("
                bla
            ")"#;

        let exp_vert = r#"//Test comment
                /*Test comment
                * hello
                */
                /* Test comment */
                #define FOO #include "bla" int bla = (1 + 1) / 2 * 10;"#;

        let expected = ShadersSources {
            vertex: transform_multi_line_str(exp_vert),
            fragment: "bla".to_string(),
            geometry: None,
            tesselation: None,
        };

        let (_, res) = shader_sources::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_pass() {
        let input = r#"pass (
                name: "MyPass"
             
                fragment ("
                    #define FOO #include "bla" int bla = (1 + 1) / 2 * 10;
                ")
                
                geometry ("
                    geom
                ")
                
                vertex ("
                    foo
                ")
                
                tesselation (
                    control ("
                        ctrl
                    ")
                    
                    evaluation ("
                        eval
                    ")
                )
            )"#;

        let expected = Pass {
            name: Some("MyPass".to_string()),
            shaders: ShadersSources {
                vertex: String::from("foo"),
                fragment: String::from(r#"#define FOO #include "bla" int bla = (1 + 1) / 2 * 10;"#),
                geometry: Some(String::from("geom")),
                tesselation: Some(Tesselation {
                    control: String::from("ctrl"),
                    evaluation: String::from("eval"),
                }),
            },
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
        let expected = Some("My Foo".to_string());

        let (_, res) = maybe_name::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_sub_shader() {
        let input = r#"
            sub_shader (
                name: "Sub Shader 1"
                
                lod: 600
                
                render_queue: Transparent(1)
                
                include ("
                    #define MY_DEFINE
                ")
                
                // bla
                /*
                 * foo
                 * bla
                 */
                passes ( // Foo
                    pass ( /*
                    * foo
                    */
                        name: "Pass1" //foo
                 
                        /*Bla*/vertex ("v1")
                        
                        fragment ("f1")
                    )
                
                    pass (
                        name: "Pass2"
                        
                        vertex ("v2")
                        
                        fragment ("f2")
                        
                        geometry ("g2")
                    )
                )
            )
        "#;

        let expected = SubShader {
            name: Some("Sub Shader 1".to_string()),
            lod: Some(600),
            render_queue: RenderQueue::Transparent(1),
            include: Some("#define MY_DEFINE".to_string()),
            passes: vec![
                Pass {
                    name: Some("Pass1".to_string()),
                    shaders: ShadersSources {
                        vertex: "v1".to_string(),
                        fragment: "f1".to_string(),
                        geometry: None,
                        tesselation: None,
                    },
                },
                Pass {
                    name: Some("Pass2".to_string()),
                    shaders: ShadersSources {
                        vertex: "v2".to_string(),
                        fragment: "f2".to_string(),
                        geometry: Some("g2".to_string()),
                        tesselation: None,
                    },
                },
            ],
        };

        let (_, res) = sub_shader::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
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
