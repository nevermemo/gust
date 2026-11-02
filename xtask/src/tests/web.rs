use std::{
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    net::{Ipv4Addr, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

use super::{
    Options, PageResult, Platform, Server, WaitEnd, chrome_candidates, content_type,
    evidence_label, feature_scope_problems, find_in_path, join_within, kill_tree,
    locked_wasm_bindgen_version, parse_chrome_version, parse_cli_version, parse_console_line,
    resolve_target, tree_enables_webgpu, wait_for_result,
};

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn scratch(name: &str) -> PathBuf {
    let directory = env::temp_dir().join(format!("gust-web-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

const ORIGIN: &str = "http://127.0.0.1:5000/";

/// A console line as Chrome writes it (format taken from the m4.g2 probe logs).
fn console(message: &str, source: &str) -> String {
    format!("[38276:35040:1101/184646.088:INFO:CONSOLE:6] \"{message}\", source: {source} (212)")
}

// ---------------------------------------------------------------------------------------------------------------
// Options

#[test]
fn options_take_every_flag() {
    let options = Options::parse(&strings(&[
        "--wasm-bindgen",
        "tools/wasm-bindgen",
        "--chrome",
        "chrome",
        "--timeout-secs",
        "7",
        "--chrome-arg",
        "--use-webgpu-adapter=swiftshader",
        "--chrome-arg",
        "--no-sandbox",
    ]))
    .unwrap();
    assert_eq!(
        options,
        Options {
            wasm_bindgen: Some(PathBuf::from("tools/wasm-bindgen")),
            chrome: Some(PathBuf::from("chrome")),
            timeout_secs: Some(7),
            chrome_args: strings(&["--use-webgpu-adapter=swiftshader", "--no-sandbox"]),
            tree_only: false,
        }
    );
    assert_eq!(options.timeout(), Duration::from_secs(7));
    assert_eq!(
        Options::parse(&[]).unwrap().timeout(),
        Duration::from_secs(90)
    );
}

#[test]
fn tree_only_is_a_mode_of_its_own() {
    assert_eq!(
        Options::parse(&strings(&["--tree-only"])).unwrap(),
        Options {
            tree_only: true,
            ..Options::default()
        }
    );
    for args in [
        &["--tree-only", "--chrome", "c"][..],
        &["--timeout-secs", "5", "--tree-only"],
        &["--tree-only", "--chrome-arg", "--x"],
        &["--wasm-bindgen", "w", "--tree-only"],
    ] {
        let error = Options::parse(&strings(args)).unwrap_err();
        assert!(
            error.contains("--tree-only takes no other flag"),
            "{args:?}: {error}"
        );
    }
    let error = Options::parse(&strings(&["--tree-only", "--tree-only"])).unwrap_err();
    assert!(error.contains("--tree-only was given twice"), "{error}");
}

#[test]
fn options_refuse_what_is_not_understood() {
    let cases: [(&[&str], &str); 7] = [
        (&["--bogus"], "unknown argument `--bogus`"),
        (&["stray"], "unknown argument `stray`"),
        (&["--chrome"], "--chrome needs a value"),
        (&["--chrome", ""], "--chrome needs a value"),
        (
            &["--chrome", "a", "--chrome", "b"],
            "--chrome was given twice",
        ),
        (&["--timeout-secs", "0"], "positive whole number, not `0`"),
        (
            &["--timeout-secs", "soon"],
            "positive whole number, not `soon`",
        ),
    ];
    for (args, expected) in cases {
        let error = Options::parse(&strings(args)).unwrap_err();
        assert!(error.contains(expected), "{args:?}: {error}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Versions

#[test]
fn the_locked_version_is_read_from_the_exact_package_name() {
    let lock = "[[package]]\nname = \"wasm-bindgen-futures\"\nversion = \"0.4.77\"\n\n[[package]]\nname = \"wasm-bindgen\"\nversion = \"0.2.127\"\ndependencies = [\n \"cfg-if\",\n]\n\n[[package]]\nname = \"wasm-bindgen-macro\"\nversion = \"0.2.127\"\n";
    assert_eq!(locked_wasm_bindgen_version(lock).unwrap(), "0.2.127");
    // A checkout with CRLF line endings reads the same.
    assert_eq!(
        locked_wasm_bindgen_version(&lock.replace('\n', "\r\n")).unwrap(),
        "0.2.127"
    );
}

#[test]
fn the_locked_version_refuses_zero_two_or_an_unreadable_entry() {
    let none = "[[package]]\nname = \"wasm-bindgen-futures\"\nversion = \"0.4.77\"\n";
    assert!(
        locked_wasm_bindgen_version(none)
            .unwrap_err()
            .contains("no wasm-bindgen package")
    );
    let two = "[[package]]\nname = \"wasm-bindgen\"\nversion = \"0.2.127\"\n\n[[package]]\nname = \"wasm-bindgen\"\nversion = \"0.2.99\"\n";
    let error = locked_wasm_bindgen_version(two).unwrap_err();
    assert!(
        error.contains("several wasm-bindgen versions (0.2.127, 0.2.99)"),
        "{error}"
    );
    let broken = "[[package]]\nname = \"wasm-bindgen\"\nsource = \"x\"\n";
    assert!(
        locked_wasm_bindgen_version(broken)
            .unwrap_err()
            .contains("without a version")
    );
}

#[test]
fn the_cli_version_is_the_second_word_of_a_wasm_bindgen_line() {
    assert_eq!(
        parse_cli_version("wasm-bindgen 0.2.127\n").as_deref(),
        Some("0.2.127")
    );
    assert_eq!(
        parse_cli_version("wasm-bindgen 0.2.127 (abcdef 2026-01-01)").as_deref(),
        Some("0.2.127")
    );
    for text in [
        "",
        "wasm-bindgen",
        "wasm-pack 0.2.127",
        "0.2.127",
        "wasm-bindgen ;rm",
    ] {
        assert_eq!(parse_cli_version(text), None, "{text:?}");
    }
}

#[test]
fn chrome_versions_are_dotted_numbers_only() {
    assert_eq!(
        parse_chrome_version("Google Chrome 156.0.8078.74 \n").as_deref(),
        Some("156.0.8078.74")
    );
    assert_eq!(
        parse_chrome_version("156.0.8078.74\r\n").as_deref(),
        Some("156.0.8078.74")
    );
    for text in ["", "Google Chrome", "156", "1.2.3.4.5", "a.b.c", "1..2"] {
        assert_eq!(parse_chrome_version(text), None, "{text:?}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tools

#[test]
fn a_tool_is_found_with_or_without_the_executable_suffix() {
    let first = scratch("path-first");
    let second = scratch("path-second");
    fs::write(
        second.join(format!("wasm-bindgen{}", env::consts::EXE_SUFFIX)),
        b"x",
    )
    .unwrap();
    let path = env::join_paths([first.as_path(), second.as_path()]).unwrap();
    assert_eq!(
        find_in_path("wasm-bindgen", &path),
        Some(second.join(format!("wasm-bindgen{}", env::consts::EXE_SUFFIX)))
    );
    assert_eq!(find_in_path("other", &path), None);
    // An empty PATH entry means the current directory to some shells; it is skipped, never searched.
    let with_empty = env::join_paths([Path::new(""), first.as_path()]).unwrap();
    assert_eq!(find_in_path("wasm-bindgen", &with_empty), None);
    let _ = fs::remove_dir_all(first);
    let _ = fs::remove_dir_all(second);
}

#[test]
fn chrome_is_looked_for_in_the_standard_places_of_each_platform() {
    let variables = |name: &str| match name {
        "ProgramFiles" => Some(OsString::from("C:/PF")),
        "LOCALAPPDATA" => Some(OsString::from("C:/Local")),
        "PATH" => Some(env::join_paths(["/usr/bin", "/opt/bin"]).unwrap()),
        _ => None,
    };
    assert_eq!(
        chrome_candidates(Platform::Windows, &variables),
        vec![
            PathBuf::from("C:/PF/Google/Chrome/Application/chrome.exe"),
            PathBuf::from("C:/Local/Google/Chrome/Application/chrome.exe"),
        ]
    );
    assert_eq!(
        chrome_candidates(Platform::Mac, &variables),
        vec![PathBuf::from(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )]
    );
    let other = chrome_candidates(Platform::Other, &variables);
    assert_eq!(other.len(), 8);
    assert_eq!(other[0], PathBuf::from("/usr/bin/google-chrome"));
    assert_eq!(other[1], PathBuf::from("/opt/bin/google-chrome"));
    assert_eq!(other[7], PathBuf::from("/opt/bin/chromium-browser"));
    assert!(chrome_candidates(Platform::Other, &|_| None).is_empty());
    // An empty variable or PATH entry would be a path relative to the current directory; it is never a candidate.
    assert!(chrome_candidates(Platform::Windows, &|_| Some(OsString::new())).is_empty());
    assert!(chrome_candidates(Platform::Other, &|_| Some(OsString::new())).is_empty());
}

#[test]
fn the_evidence_label_names_the_browser_and_marks_software_adapters() {
    assert_eq!(
        evidence_label(Some("156.0.8078.74"), "nvidia blackwell"),
        "Gpu (WebGPU, Chrome 156.0.8078.74, nvidia blackwell)"
    );
    assert_eq!(
        evidence_label(Some("156.0.8078.74"), "google swiftshader"),
        "Gpu (software, WebGPU, Chrome 156.0.8078.74, google swiftshader)"
    );
    assert_eq!(
        evidence_label(None, "amd rdna-3"),
        "Gpu (WebGPU, Chrome unknown version, amd rdna-3)"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The result line

#[test]
fn a_console_line_with_the_ok_message_from_our_server_is_a_result() {
    let line = console(
        "GUST_OK nvidia blackwell",
        "http://127.0.0.1:5000/web_vector_add.js",
    );
    assert_eq!(
        parse_console_line(&line, ORIGIN),
        Some(PageResult::Ok {
            label: "nvidia blackwell".into()
        })
    );
    let line = console(
        "GUST_FAIL read-back failed: \"map\" error",
        "http://127.0.0.1:5000/",
    );
    assert_eq!(
        parse_console_line(&line, ORIGIN),
        Some(PageResult::Fail {
            reason: "read-back failed: \"map\" error".into()
        })
    );
}

#[test]
fn only_an_exact_result_message_from_our_own_origin_counts() {
    let source = "http://127.0.0.1:5000/index.html";
    // Same text, wrong origin: another port that starts with the same digits, another host, a file.
    for foreign in [
        "http://127.0.0.1:50001/index.html",
        "http://localhost:5000/index.html",
        "file:///C:/page.html",
        "wasm://wasm/web.wasm-0018cdaa",
    ] {
        assert_eq!(
            parse_console_line(&console("GUST_OK nvidia blackwell", foreign), ORIGIN),
            None,
            "{foreign}"
        );
    }
    // Same origin, not a result message.
    for message in [
        "GUST_OKAY nvidia blackwell",
        "GUST_OK",
        "GUST_OK   ",
        "GUST_FAILED x",
        "xGUST_OK nvidia blackwell",
        " GUST_OK nvidia blackwell",
        "PANIC panicked at library/std/src/sys/time/unsupported.rs:13:9",
        "The powerPreference option is currently ignored when calling requestAdapter() on Windows.",
    ] {
        assert_eq!(
            parse_console_line(&console(message, source), ORIGIN),
            None,
            "{message}"
        );
    }
    // Not a console line at all, even though the text is a result.
    for line in [
        "GUST_OK nvidia blackwell",
        "[1:2:1101/1.5:ERROR:chrome\\x.cc:1] \"GUST_OK a b\", source: http://127.0.0.1:5000/ (1)",
        "[1:2:1101/1.5:INFO:CONSOLE:6] GUST_OK a b, source: http://127.0.0.1:5000/ (1)",
        "[1:2:1101/1.5:INFO:CONSOLE:6] \"GUST_OK a b\" source: http://127.0.0.1:5000/ (1)",
        "",
    ] {
        assert_eq!(parse_console_line(line, ORIGIN), None, "{line}");
    }
}

fn channel_with(lines: &[String]) -> (mpsc::Sender<String>, mpsc::Receiver<String>) {
    let (sender, receiver) = mpsc::channel();
    for line in lines {
        sender.send(line.clone()).unwrap();
    }
    (sender, receiver)
}

#[test]
fn the_wait_returns_the_first_result_and_keeps_the_lines_before_it() {
    let (_sender, receiver) = channel_with(&[
        "[1:2:3:ERROR:noise] x".to_owned(),
        console("GUST_FAIL boom", "http://127.0.0.1:5000/a.js"),
        console("GUST_OK nvidia blackwell", "http://127.0.0.1:5000/a.js"),
    ]);
    let (result, end, tail) = wait_for_result(
        &receiver,
        Instant::now() + Duration::from_secs(30),
        ORIGIN,
        &mut || true,
    );
    assert_eq!(end, WaitEnd::Result);
    assert_eq!(
        result,
        Some(PageResult::Fail {
            reason: "boom".into()
        })
    );
    assert_eq!(tail.len(), 2);
}

#[test]
fn the_wait_gives_up_at_the_deadline_when_nothing_is_logged() {
    // The sender stays alive and Chrome counts as running, like a page that never logs.
    let (_sender, receiver) = channel_with(&[
        "[1:2:3:INFO:CONSOLE:1] \"hello\", source: http://127.0.0.1:5000/ (1)".to_owned(),
    ]);
    let started = Instant::now();
    let (result, end, tail) = wait_for_result(
        &receiver,
        started + Duration::from_millis(400),
        ORIGIN,
        &mut || true,
    );
    assert_eq!((result, end), (None, WaitEnd::TimedOut));
    assert!(started.elapsed() >= Duration::from_millis(400));
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(tail.len(), 1);
}

#[test]
fn the_wait_notices_a_browser_that_ended_and_still_reads_what_it_wrote() {
    // Stream ended.
    let (sender, receiver) = channel_with(&["[1:2:3:ERROR:noise] x".to_owned()]);
    drop(sender);
    let deadline = Instant::now() + Duration::from_secs(30);
    let (result, end, _) = wait_for_result(&receiver, deadline, ORIGIN, &mut || true);
    assert_eq!((result, end), (None, WaitEnd::BrowserGone));
    // Process gone but the pipe still open (a grandchild holds it), with no result written.
    let (_sender, receiver) = channel_with(&[]);
    let (result, end, _) = wait_for_result(&receiver, deadline, ORIGIN, &mut || false);
    assert_eq!((result, end), (None, WaitEnd::BrowserGone));
}

// ---------------------------------------------------------------------------------------------------------------
// The server

#[test]
fn only_plain_file_names_are_served() {
    let cases = [
        ("/", Some("index.html")),
        ("/index.html", Some("index.html")),
        ("/web_vector_add_bg.wasm", Some("web_vector_add_bg.wasm")),
        ("/a.js?v=1", Some("a.js")),
        ("/a.js#frag", Some("a.js")),
        ("/?x", Some("index.html")),
        ("/..", None),
        ("/../Cargo.toml", None),
        ("/sub/a.js", None),
        ("/sub/", None),
        ("/%2e%2e/x", None),
        ("/a%20b", None),
        ("/.hidden", None),
        ("/a\\b", None),
        ("/C:/x", None),
        ("/a b", None),
        ("a.js", None),
        ("", None),
        ("http://evil/a.js", None),
        ("*", None),
    ];
    for (target, expected) in cases {
        assert_eq!(resolve_target(target).as_deref(), expected, "{target:?}");
    }
}

#[test]
fn files_get_the_content_type_their_extension_needs() {
    assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
    assert_eq!(content_type("a.js"), "text/javascript; charset=utf-8");
    assert_eq!(content_type("a_bg.wasm"), "application/wasm");
    assert_eq!(content_type("m.JSON"), "application/json");
    assert_eq!(content_type("x"), "application/octet-stream");
}

fn request(server: &Server, raw: &str) -> (String, String, Vec<u8>) {
    let mut stream = TcpStream::connect(server.addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream.write_all(raw.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("a complete response head");
    let head = String::from_utf8(response[..split].to_vec()).unwrap();
    let status = head.lines().next().unwrap().to_owned();
    (status, head, response[split + 4..].to_vec())
}

#[test]
fn the_server_binds_loopback_and_serves_only_get_of_plain_files() {
    let directory = scratch("server");
    fs::write(directory.join("index.html"), "<p>page</p>").unwrap();
    fs::write(directory.join("a_bg.wasm"), [0u8, 0x61, 0x73, 0x6d]).unwrap();
    fs::write(directory.join(".hidden"), "secret").unwrap();
    fs::create_dir_all(directory.join("sub")).unwrap();
    fs::write(directory.join("sub").join("x.js"), "x").unwrap();
    let mut server = Server::start(directory.clone()).unwrap();
    assert_eq!(server.addr().ip(), Ipv4Addr::LOCALHOST);
    assert_ne!(server.addr().port(), 0);
    assert_eq!(
        server.origin(),
        format!("http://127.0.0.1:{}/", server.addr().port())
    );

    let (status, head, body) = request(&server, "GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert!(
        head.contains("Content-Type: text/html; charset=utf-8"),
        "{head}"
    );
    assert!(head.contains("Content-Length: 11"), "{head}");
    assert_eq!(body, b"<p>page</p>");

    let (status, head, body) = request(&server, "GET /a_bg.wasm?v=2 HTTP/1.1\r\n\r\n");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert!(head.contains("Content-Type: application/wasm"), "{head}");
    assert_eq!(body, [0u8, 0x61, 0x73, 0x6d]);

    for target in [
        "/missing.js",
        "/.hidden",
        "/sub/x.js",
        "/sub",
        "/../index.html",
        "/%2e%2e/index.html",
    ] {
        let (status, _, body) = request(&server, &format!("GET {target} HTTP/1.1\r\n\r\n"));
        assert_eq!(status, "HTTP/1.1 404 Not Found", "{target}");
        assert_eq!(body, b"not found", "{target}");
    }
    let (status, _, _) = request(&server, "POST / HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    assert_eq!(status, "HTTP/1.1 405 Method Not Allowed");

    assert_eq!(
        server.served(),
        strings(&[
            "GET /",
            "GET /a_bg.wasm",
            "GET /missing.js",
            "GET /.hidden",
            "GET /sub/x.js",
            "GET /sub",
            "GET /../index.html",
            "GET /%2e%2e/index.html",
            "POST /",
        ])
    );

    let address = server.addr();
    server.stop();
    assert!(
        TcpStream::connect_timeout(&address, Duration::from_secs(2)).is_err(),
        "the listener is closed after stop"
    );
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn an_idle_connection_does_not_block_other_requests() {
    // Chrome opens speculative connections that send nothing.
    let directory = scratch("server-idle");
    fs::write(directory.join("index.html"), "ok").unwrap();
    let mut server = Server::start(directory.clone()).unwrap();
    let _idle = TcpStream::connect(server.addr()).unwrap();
    let started = Instant::now();
    let (status, _, body) = request(&server, "GET / HTTP/1.1\r\n\r\n");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(body, b"ok");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );
    server.stop();
    let _ = fs::remove_dir_all(directory);
}

// ---------------------------------------------------------------------------------------------------------------
// Killing Chrome

#[test]
fn kill_tree_ends_a_running_child_and_returns() {
    let mut command = if cfg!(windows) {
        let mut command = Command::new("powershell");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Start-Sleep -Seconds 120",
        ]);
        command
    } else {
        let mut command = Command::new("sleep");
        command.arg("120");
        command
    };
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    assert!(child.try_wait().unwrap().is_none(), "the sleeper runs");
    let started = Instant::now();
    kill_tree(&mut child);
    assert!(started.elapsed() < Duration::from_secs(30));
    assert!(child.try_wait().unwrap().is_some(), "the child is gone");
}

// ---------------------------------------------------------------------------------------------------------------
// The feature guard

/// The shape of `cargo tree --locked --workspace -e features -i wgpu` (box drawing; `--charset ascii` differs).
const HOST_TREE: &str = "wgpu v30.0.1\n├── wgpu feature \"std\"\n│   └── gust-wgpu v0.1.0 (path)\n├── wgpu feature \"vulkan\"\n│   └── gust-wgpu v0.1.0 (path) (*)\n└── wgpu feature \"wgsl\"\n    └── gust-wgpu v0.1.0 (path) (*)\n";

const WASM_TREE: &str = "wgpu v30.0.1\n├── wgpu feature \"std\"\n│   └── gust-wgpu v0.1.0 (path)\n├── wgpu feature \"webgpu\"\n│   └── gust-wgpu v0.1.0 (path) (*)\n└── wgpu feature \"wgsl\"\n    └── gust-wgpu v0.1.0 (path) (*)\n";

#[test]
fn the_guard_finds_the_webgpu_feature_line_in_either_drawing() {
    assert!(!tree_enables_webgpu(HOST_TREE));
    assert!(tree_enables_webgpu(WASM_TREE));
    assert!(tree_enables_webgpu(
        "wgpu v30.0.1\n|-- wgpu feature \"webgpu\"\n"
    ));
    assert!(tree_enables_webgpu(
        "wgpu v30.0.1\n`-- wgpu feature \"webgpu\"\r\n"
    ));
    assert!(tree_enables_webgpu(
        "├── wgpu feature \"webgpu\" (command-line)\n"
    ));
    // Other crates' features, other wgpu features and mentions in a path are not the feature.
    for tree in [
        "wgpu-types feature \"webgpu\"\n",
        "wgpu feature \"webgpu-extra\"\n",
        "└── gust-wgpu v0.1.0 (C:/x/wgpu feature \"webgpu\")\n",
        "",
    ] {
        assert!(!tree_enables_webgpu(tree), "{tree:?}");
    }
}

#[test]
fn the_guard_passes_only_for_webgpu_on_wasm32_alone() {
    assert_eq!(
        feature_scope_problems(HOST_TREE, WASM_TREE),
        Vec::<String>::new()
    );
    // Mutation d: the host build gains the feature.
    let gained = feature_scope_problems(WASM_TREE, WASM_TREE);
    assert_eq!(gained.len(), 1, "{gained:?}");
    assert!(gained[0].contains("a host build enables the wgpu `webgpu` feature"));
    // The control: no wasm32 feature means the browser has no backend, and the guard must not pass vacuously.
    let missing = feature_scope_problems(HOST_TREE, HOST_TREE);
    assert_eq!(missing.len(), 1, "{missing:?}");
    assert!(missing[0].contains("wasm32 build does not enable"));
    // A host tree without wgpu proves nothing.
    let empty = feature_scope_problems("", WASM_TREE);
    assert_eq!(empty.len(), 1, "{empty:?}");
    assert!(empty[0].contains("does not show wgpu at all"));
}

// ---------------------------------------------------------------------------------------------------------------
// Bounded joins

#[test]
fn join_within_waits_for_a_thread_that_ends_and_gives_up_on_one_that_does_not() {
    let quick = std::thread::spawn(|| 7);
    assert!(join_within(quick, Duration::from_secs(5)));

    let (release, hold) = mpsc::channel::<()>();
    let stuck = std::thread::spawn(move || {
        let _ = hold.recv();
    });
    let started = Instant::now();
    assert!(!join_within(stuck, Duration::from_millis(150)));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
    // Let the left-behind thread end.
    drop(release);
}

// ---------------------------------------------------------------------------------------------------------------
// The committed artifacts of the example

/// The five committed copies of the kernels fixture's artifact, the web example's among them (its own test below also
/// compares its manifest and file set). Each holds the kernel's WGSL and reflection under the same file names; the
/// manifests of the others are not compared here, because the crate fixtures and the UI sets keep a manifest of their
/// own.
const ARTIFACT_COPIES: [&str; 5] = [
    "examples/web-vector-add/gust-artifacts",
    "tests/ui/profile-native-artifact/gust-artifacts",
    "tests/ui/profile-native-no-manifest/gust-artifacts",
    "crates/gust/tests/fixtures/precompiled",
    "crates/gust-wgpu/tests/fixtures/precompiled",
];

#[test]
fn all_five_committed_copies_of_the_precompiled_artifact_are_byte_copies_of_the_kernels_fixture() {
    // m4.r re-blessed every set and only the web copy was byte-checked, so a copy left stale passed
    // `verify_artifact` (which reads its own pair of files) until a test happened to use it. One source, five copies:
    // a stale one fails here and names itself.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let source = root.join("tests/precompiled/kernels/gust-artifacts");
    let mut artifact_files: Vec<String> = fs::read_dir(&source)
        .unwrap_or_else(|error| panic!("{}: {error}", source.display()))
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name != "manifest.json")
        .collect();
    artifact_files.sort();
    assert_eq!(
        artifact_files.len(),
        2,
        "the kernels fixture holds one WGSL and one reflection file: {artifact_files:?}"
    );
    for copy in ARTIFACT_COPIES {
        for name in &artifact_files {
            let path = root.join(copy).join(name);
            let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{copy}/{name}: {error}"));
            assert!(
                bytes == fs::read(source.join(name)).unwrap(),
                "{copy}/{name} differs from tests/precompiled/kernels/gust-artifacts/{name}: re-copy it from there"
            );
        }
    }
}

#[test]
fn the_web_example_artifacts_are_a_byte_copy_of_the_precompiled_fixtures() {
    // The example cannot regenerate them (a browser build never runs Slang, and on the host the crate is empty), so the
    // copy has to follow the fixture: m4.r's re-bless of tests/precompiled/kernels/gust-artifacts must re-copy it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let source = root.join("tests/precompiled/kernels/gust-artifacts");
    let copy = root.join("examples/web-vector-add/gust-artifacts");
    let names = |directory: &Path| {
        let mut names: Vec<String> = fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    let expected = names(&source);
    assert!(!expected.is_empty(), "no artifacts in {}", source.display());
    assert_eq!(
        names(&copy),
        expected,
        "examples/web-vector-add/gust-artifacts must hold exactly the files of tests/precompiled/kernels/gust-artifacts"
    );
    for name in expected {
        assert_eq!(
            fs::read(copy.join(&name)).unwrap(),
            fs::read(source.join(&name)).unwrap(),
            "{name} differs: re-copy tests/precompiled/kernels/gust-artifacts/* into examples/web-vector-add/gust-artifacts/"
        );
    }
}
