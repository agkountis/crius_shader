use crate::*;

use nom::{
    branch::{alt, permutation},
    bytes::complete::{escaped, is_not, tag, take_until},
    character::complete::{alphanumeric1, char, digit1, multispace0, one_of},
    combinator::{cut, map, map_res, opt, value, verify},
    error::{context, ParseError, VerboseError},
    multi::{many1, separated_list},
    number::complete::float,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};

fn ws<'a, F: 'a, O, E>(inner: F) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(multispace0, inner, multispace0)
}

fn parse_esc_str<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    escaped(alphanumeric1, '\\', one_of("\"n\\"))(input)
}

pub fn string<'a, E>(i: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str>,
{
    context(
        "string",
        preceded(char('\"'), cut(terminated(take_until(r#"""#), char('\"')))),
    )(i)
}

fn boolean<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, bool, E> {
    let parse_true = value(true, tag("true"));
    let parse_false = value(false, tag("false"));
    alt((parse_true, parse_false))(input)
}

fn float2<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str>,
{
    let vec2 = preceded(
        ws(tag("(")),
        cut(terminated(
            verify(separated_list(ws(tag(",")), float), |list: &[f32]| {
                list.len() == 2
            }),
            ws(tag(")")),
        )),
    );

    context("float2", vec2)(input)
}

fn float3<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str>,
{
    let vec3 = preceded(
        ws(tag("(")),
        cut(terminated(
            verify(separated_list(ws(tag(",")), float), |list: &[f32]| {
                list.len() == 3
            }),
            ws(tag(")")),
        )),
    );

    context("float3", vec3)(input)
}

fn float4<'a, E>(input: &'a str) -> IResult<&'a str, Vec<f32>, E>
where
    E: 'a + ParseError<&'a str>,
{
    let vec4 = preceded(
        ws(tag("(")),
        cut(terminated(
            verify(separated_list(ws(tag(",")), float), |list: &[f32]| {
                list.len() == 4
            }),
            ws(tag(")")),
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
        pair(tag("//"), is_not("\n\r")),
    )(i)
}

fn multi_line_comment<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, (), E> {
    value(
        (), // Output is thrown away.
        tuple((tag("/*"), take_until("*/"), tag("*/"))),
    )(i)
}

fn comment<'a, E: 'a + ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (), E> {
    let a = ws(single_line_comment);
    let b = ws(multi_line_comment);
    alt((a, b))(input)
}

fn parse_block<'a, F: 'a, O, E>(
    open_tag: &'a str,
    inner: F,
    close_tag: &'a str,
) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    E: 'a + ParseError<&'a str>,
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    context(
        "parse_block",
        preceded(
            ws(tag(open_tag)),
            cut(terminated(ws(inner), ws(tag(close_tag)))),
        ),
    )
}

fn block<'a, F, O, E>(parser: F) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    E: 'a + ParseError<&'a str>,
    F: 'a + Fn(&'a str) -> IResult<&'a str, O, E>,
{
    context("block", parse_block("(", parser, ")"))
}

fn string_block<'a, F: 'a, O, E>(inner: F) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    E: 'a + ParseError<&'a str>,
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    context("string_block", parse_block(r#"(""#, inner, r#"")"#))
}

fn named<'a, F: 'a, O, E>(name: &'a str, parser: F) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    E: 'a + ParseError<&'a str>,
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    context("named", preceded(ws(tag(name)), parser))
}

fn string_block_contents<'a, E>(input: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: 'a + ParseError<&'a str>,
{
    take_until(r#"")"#)(input)
}

fn trim_ws<'a, F, E>(parser: F) -> impl Fn(&'a str) -> IResult<&'a str, String, E>
where
    E: 'a + ParseError<&'a str>,
    F: 'a + Fn(&'a str) -> IResult<&'a str, &'a str, E>,
{
    context(
        "trimmed_string",
        map(parser, |s| {
            s.lines()
                .into_iter()
                .fold(String::new(), |acc, s| acc + s.trim())
        }),
    )
}

fn named_string_block<'a, E>(name: &'a str) -> impl Fn(&'a str) -> IResult<&'a str, String, E>
where
    E: 'a + ParseError<&'a str>,
{
    named(name, string_block(trim_ws(string_block_contents)))
}

fn optional_shader_source<'a, E>(
    input: &'a str,
) -> IResult<&'a str, Option<OptionalShaderSource>, E>
where
    E: 'a + ParseError<&'a str>,
{
    let tesselation_ctrl = named_string_block(TESSELATION_CONTROL_SHADER_TAG);
    let tesselation_eval = named_string_block(TESSELATION_EVALUATION_SHADER_TAG);
    let geometry = named_string_block(GEOMETRY_SHADER_TAG);
    let tesselation_block = named(
        TESSELATION_BLOCK_TAG,
        block(permutation((tesselation_ctrl, tesselation_eval))),
    );

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
    E: 'a + ParseError<&'a str>,
{
    map(
        permutation((
            named_string_block(VERTEX_SHADER_TAG),
            named_string_block(FRAGMENT_SHADER_TAG),
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
    )(input)
}

pub fn separated_value<'a, I, O1, O2, O3, E, F, G, H>(
    before: F,
    separator: G,
    value_parser: H,
) -> impl Fn(I) -> IResult<I, O3, E>
where
    E: 'a + ParseError<I>,
    F: Fn(I) -> IResult<I, O1, E>,
    G: Fn(I) -> IResult<I, O2, E>,
    H: Fn(I) -> IResult<I, O3, E>,
{
    preceded(before, preceded(separator, value_parser))
}

pub fn optional_name<'a, E>(input: &'a str) -> IResult<&'a str, Option<String>, E>
where
    E: 'a + ParseError<&'a str>,
{
    opt(map(
        separated_value(ws(tag("name")), ws(tag(":")), ws(string)),
        String::from,
    ))(input)
}

pub fn optional_lod<'a, E>(input: &'a str) -> IResult<&'a str, Option<u32>, E>
where
    E: 'a + ParseError<&'a str>,
{
    opt(separated_value(
        ws(tag("lod")),
        ws(tag(":")),
        map_res(ws(parse_esc_str), |a| a.parse::<u32>()),
    ))(input)
}

pub fn pass<'a, E>(input: &'a str) -> IResult<&'a str, Pass, E>
where
    E: 'a + ParseError<&'a str>,
{
    let parse_pass = named("pass", block(tuple((optional_name, shader_sources))));
    map(parse_pass, |(name, shaders)| Pass { name, shaders })(input)
}

pub fn render_queue<'a, E>(input: &'a str) -> IResult<&'a str, RenderQueue, E>
where
    E: 'a + ParseError<&'a str>,
{
    let parse_val = alt((
        value(RenderQueue::Opaque, ws(tag("Opaque"))),
        map(
            preceded(
                tag("Transparent"),
                delimited(
                    ws(tag("(")),
                    map_res(digit1, str::parse::<u32>),
                    ws(tag(")")),
                ),
            ),
            |num| RenderQueue::Transparent(num),
        ),
    ));
    preceded(
        ws(tag("render_queue")),
        preceded(
            ws(tag(":")),
            ws(delimited(ws(tag("\"")), parse_val, ws(tag("\"")))),
        ),
    )(input)
}

pub fn sub_shader<'a, E>(input: &'a str) -> IResult<&'a str, SubShader, E>
where
    E: 'a + ParseError<&'a str>,
{
    let optional_lod = opt(separated_value(
        ws(tag("lod")),
        ws(tag(":")),
        map_res(ws(parse_esc_str), |a| a.parse::<u32>()),
    ));

    let optional_include = opt(named_string_block("include"));
    let passes = many1(pass);

    let contents = map(
        tuple((
            optional_name,
            optional_lod,
            render_queue,
            optional_include,
            passes,
        )),
        |(name, lod, render_queue, include, passes)| SubShader {
            name,
            lod,
            render_queue,
            include,
            passes,
        },
    );

    named("sub_shader", block(contents))(input)
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
        let input = "/* hello world\nMulti line*/\nFoo";
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
            ")
        "#;

        let exp1 = "layout(location = 0) in vec3 attrPos;\
        struct Foo {vec3 bla;};void main(){#define FOO #include \"bla\" \
        int bla = (1 + 1) / 2 * 10;}";

        fn f<'a>(tag: &'a str, input: &'a str) -> IResult<&'a str, String, VerboseError<&'a str>> {
            named_string_block::<VerboseError<&str>>(tag)(input)
        };

        match f(VERTEX_SHADER_TAG, input) {
            Err(nom::Err::Failure(e)) | Err(nom::Err::Error(e)) => {
                println!("{}", convert_error(input, e))
            }
            Ok((i, res)) => {
                println!("Result: {}", res);
                println!("Remaining: {}", i);
                assert_eq!(exp1, res)
            }
            _ => {}
        }
    }

    #[test]
    fn test_shaders() {
        let input = r#"
            vertex ("
                #define FOO #include "bla" int bla = (1 + 1) / 2 * 10;
            ")
            
            fragment ("
                bla
            ")"#;

        //#define FOO #include "bla" int bla = (1 + 1) / 2 * 10;

        let expected = ShadersSources {
            vertex: r#"#define FOO #include "bla" int bla = (1 + 1) / 2 * 10;"#.to_string(),
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
                vertex: "foo".to_string(),
                fragment: r#"#define FOO #include "bla" int bla = (1 + 1) / 2 * 10;"#.to_string(),
                geometry: Some("geom".to_string()),
                tesselation: Some(Tesselation {
                    control: "ctrl".to_string(),
                    evaluation: "eval".to_string(),
                }),
            },
        };

        let (_, res) = pass::<VerboseError<&str>>(input).unwrap();
        assert_eq!(res, expected)
    }

    #[test]
    fn test_render_queue() {
        let inputs = vec![
            r#"render_queue: "Opaque""#,
            r#"render_queue: "Transparent(0)""#,
            r#"render_queue: "Transparent(10)""#,
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

        let (_, res) = optional_name::<VerboseError<&str>>(input).unwrap();
        assert_eq!(expected, res)
    }

    #[test]
    fn test_sub_shader() {
        let input = r#"
            sub_shader (
                name: "Sub Shader 1"
                
                lod: 600
                
                render_queue: "Transparent(1)"
                
                include ("
                    #define MY_DEFINE
                ")
                
                pass (
                    name: "Pass1"
                 
                    vertex ("
                        v1
                    ")
                    
                    fragment ("
                        f1
                    ")
                )
                
                pass (
                    name: "Pass2"
                    
                    vertex ("
                        v2
                    ")
                    
                    fragment ("
                        f2
                    ")
                    
                    geometry ("
                        g2
                    ")
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
}
