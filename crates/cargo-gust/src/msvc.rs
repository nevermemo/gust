//! Finding a C++ toolchain on Windows when `cl` is not on `PATH`: run `vcvars64.bat` in `cmd.exe` and capture the
//! environment it produces (moved from `xtask/src/reflect.rs` in slice m4.f, with its tests).
//!
//! Only the helper build uses this. The functions are pure where they can be (the `set` dump parser, the command
//! line), so the tests run on every platform; the ones that start `cmd.exe` run on Windows only.

use std::{
    env,
    path::{Path, PathBuf},
};

pub(crate) fn find_vcvars64() -> Result<PathBuf, String> {
    let program_files = env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .ok_or("ProgramFiles(x86) is not set")?;
    let vswhere = program_files.join("Microsoft Visual Studio/Installer/vswhere.exe");
    if !vswhere.is_file() {
        return Err("Use an x64 MSVC developer shell or install the MSVC C++ build tools.".into());
    }
    let output = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err("An x64 MSVC C++ toolchain is required.".into());
    }
    let install = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let vcvars = PathBuf::from(install).join("VC/Auxiliary/Build/vcvars64.bat");
    if vcvars.is_file() {
        Ok(vcvars)
    } else {
        Err("vcvars64.bat was not found in the Visual Studio installation".into())
    }
}

/// The command line (after the program name) that makes `cmd.exe` run `vcvars64.bat` and then
/// print the resulting environment with `set`.
///
/// Switches, in order:
/// - `/D` skips the registry AutoRun commands, whose output would contaminate the capture.
/// - `/U` makes cmd write its internal commands' output (`set`) as UTF-16LE, so non-ASCII
///   directories in PATH, INCLUDE, LIB or TEMP survive; the parent decodes it explicitly.
/// - `/V:OFF` disables delayed expansion, so a `!name!` inside an accepted path stays literal.
/// - `/S` makes cmd strip exactly the outer pair of quotes and leave the inner ones alone, which
///   is what keeps a path with spaces and parentheses (`Program Files (x86)`) intact.
///
/// The batch file's own output is redirected to stderr (`1>&2`) so stdout carries only the `set`
/// dump and stays decodable as pure UTF-16LE; stderr is kept for the failure message.
///
/// The string must reach cmd verbatim: `Command::arg` would escape the inner quotes as `\"`,
/// which cmd does not understand. Characters that cannot be quoted safely in a cmd line are
/// rejected.
fn vcvars_capture_command_line(vcvars: &Path) -> Result<String, String> {
    let path = vcvars
        .to_str()
        .ok_or("non-utf8 vcvars64.bat path")?
        .replace('/', "\\");
    if path.contains(['"', '%', '\r', '\n']) {
        return Err(format!(
            "vcvars64.bat path contains a character cmd.exe cannot quote safely: {path}"
        ));
    }
    Ok(format!("/D /U /V:OFF /S /C \"\"{path}\" 1>&2 && set\""))
}

/// Decodes UTF-16LE bytes (what `cmd /U` writes), dropping a leading byte order mark.
fn decode_utf16le(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() & 1 == 1 {
        return Err(format!(
            "the environment dump is not UTF-16LE (odd length {})",
            bytes.len()
        ));
    }
    let text = String::from_utf16_lossy(&utf16le_units(bytes));
    Ok(text.strip_prefix('\u{feff}').unwrap_or(&text).to_owned())
}

/// Pairs up little-endian bytes into UTF-16 code units; a trailing odd byte is ignored.
fn utf16le_units(bytes: &[u8]) -> Vec<u16> {
    bytes
        .iter()
        .step_by(2)
        .zip(bytes.iter().skip(1).step_by(2))
        .map(|(low, high)| u16::from_le_bytes([*low, *high]))
        .collect()
}

/// Decodes diagnostic output of unknown encoding: batch `echo` is UTF-16LE under `cmd /U`, other
/// programs write bytes in their own code page. Mostly-NUL bytes mean UTF-16LE.
fn decode_diagnostic(bytes: &[u8]) -> String {
    let zeros = bytes.iter().filter(|byte| **byte == 0).count();
    if bytes.len() >= 2 && zeros * 4 >= bytes.len() {
        String::from_utf16_lossy(&utf16le_units(bytes))
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// The last `count` non-empty lines of the diagnostic output, joined with ` | `.
fn diagnostic_tail(bytes: &[u8], count: usize) -> String {
    let text = decode_diagnostic(bytes);
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines[lines.len().saturating_sub(count)..].join(" | ")
}

/// A variable name this parser accepts: non-empty, alphanumeric plus a few punctuation marks
/// (`_`, parentheses as in `ProgramFiles(x86)`, and similar). Anything else is not a variable line.
fn is_variable_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || "_()-.$#@{}[]~".contains(c))
}

/// Parses the output of `set` into `(name, value)` pairs.
///
/// `set` prints a multi-line value raw, so a line is only trusted when it looks like `NAME=value`
/// with a valid name. A line that does not (a continuation of a multi-line value) drops the
/// variable before it. A continuation line that itself looks like `NAME=value` cannot be told
/// apart from a real variable, so a name that appears twice (cmd names are case-insensitive, so
/// real duplicates cannot exist) is an error rather than a silent override; this is what stops an
/// inherited value from forging a second PATH, INCLUDE or LIB. The hidden per-drive entries cmd
/// prints (`=C:=C:\dir`) are skipped.
///
/// The duplicate rule alone does not catch a forged name that vcvars does not set (for example
/// `CL`). That case is refused before the capture runs, by [`ambiguous_inherited_variable`].
fn parse_env_dump(dump: &str) -> Result<Vec<(String, String)>, String> {
    let mut variables: Vec<(String, String)> = Vec::new();
    let mut last_is_open = false;
    for line in dump.lines() {
        if line.starts_with('=') {
            continue;
        }
        match line.split_once('=') {
            Some((name, value)) if is_variable_name(name) => {
                variables.push((name.to_owned(), value.to_owned()));
                last_is_open = true;
            }
            _ => {
                if last_is_open {
                    variables.pop();
                    last_is_open = false;
                }
            }
        }
    }
    for (index, (name, _)) in variables.iter().enumerate() {
        if variables[..index]
            .iter()
            .any(|(earlier, _)| earlier.eq_ignore_ascii_case(name))
        {
            return Err(format!(
                "the environment dump defines `{name}` more than once; an inherited multi-line value may be forging it"
            ));
        }
    }
    Ok(variables)
}

/// The name of the first inherited variable whose value makes the `set` dump ambiguous, if any.
///
/// `set` prints a multi-line value raw. A continuation line shaped like `NAME=value` (or like a
/// hidden `=C:=...` entry) cannot be told from a real variable in the dump, so such a value could
/// forge a variable that vcvars never set, for example `CL`, which `cl.exe` reads as extra
/// command-line options. A multi-line value whose continuation lines are plain text is harmless:
/// [`parse_env_dump`] drops the variable together with those lines.
fn ambiguous_inherited_variable(environment: &[(String, String)]) -> Option<&str> {
    environment
        .iter()
        .find(|(_, value)| {
            value.lines().skip(1).any(|line| {
                line.starts_with('=')
                    || line
                        .split_once('=')
                        .is_some_and(|(name, _)| is_variable_name(name))
            })
        })
        .map(|(name, _)| name.as_str())
}

/// The environment the capture process will inherit: the current process environment with the
/// command's own `env`/`env_remove` changes applied. Names compare case-insensitively, like cmd's.
fn inherited_environment(command: &std::process::Command) -> Vec<(String, String)> {
    let mut environment: Vec<(String, String)> = env::vars_os()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect();
    for (name, value) in command.get_envs() {
        let name = name.to_string_lossy().into_owned();
        environment.retain(|(existing, _)| !existing.eq_ignore_ascii_case(&name));
        if let Some(value) = value {
            environment.push((name, value.to_string_lossy().into_owned()));
        }
    }
    environment
}

/// Looks for `program` in the `;`-separated (platform list separator) directories of `path`.
fn find_in_path(program: &str, path: &str) -> Option<PathBuf> {
    env::split_paths(path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

/// The value of PATH in a captured environment, whatever its casing (`Path`, `PATH`, `path`).
fn environment_path(environment: &[(String, String)]) -> Option<&str> {
    environment
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("path"))
        .map(|(_, value)| value.as_str())
}

#[cfg(windows)]
fn set_raw_command_line(command: &mut std::process::Command, line: &str) {
    use std::os::windows::process::CommandExt;
    command.raw_arg(line);
}

#[cfg(not(windows))]
fn set_raw_command_line(command: &mut std::process::Command, line: &str) {
    command.arg(line);
}

/// The `cmd.exe` process that runs `vcvars64.bat` and dumps the environment (not yet started).
fn vcvars_capture_command(vcvars: &Path) -> Result<std::process::Command, String> {
    let mut command = std::process::Command::new("cmd");
    set_raw_command_line(&mut command, &vcvars_capture_command_line(vcvars)?);
    Ok(command)
}

/// Runs a prepared capture command and decodes the environment it printed.
fn capture_environment_from(
    mut command: std::process::Command,
) -> Result<Vec<(String, String)>, String> {
    if let Some(name) = ambiguous_inherited_variable(&inherited_environment(&command)) {
        return Err(format!(
            "the inherited environment variable `{name}` holds a multi-line value with a line shaped like a variable assignment; the vcvars environment dump could not tell it from a real variable. Unset `{name}` and retry"
        ));
    }
    let output = command
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|error| format!("could not launch cmd: {error}"))?;
    if !output.status.success() {
        let mut detail = diagnostic_tail(&output.stderr, 5);
        if detail.is_empty() {
            detail = diagnostic_tail(&output.stdout, 5);
        }
        return Err(format!(
            "vcvars64.bat failed (exit {}): {detail}",
            output.status.code().unwrap_or(-1)
        ));
    }
    parse_env_dump(&decode_utf16le(&output.stdout)?)
}

/// Runs `vcvars64.bat` once in `cmd.exe` and returns the environment it produced.
pub(crate) fn capture_vcvars_environment(vcvars: &Path) -> Result<Vec<(String, String)>, String> {
    capture_environment_from(vcvars_capture_command(vcvars)?)
}

/// Runs `cl.exe` directly (no shell) with the environment captured from `vcvars64.bat`.
pub(crate) fn run_cl_in_environment(
    environment: &[(String, String)],
    args: &[String],
) -> Result<(), String> {
    let path = environment_path(environment).ok_or("vcvars64.bat did not produce a PATH")?;
    let cl = find_in_path("cl.exe", path)
        .ok_or("cl.exe was not found on the PATH that vcvars64.bat produced")?;
    let status = std::process::Command::new(&cl)
        .args(args)
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|error| format!("could not launch {}: {error}", cl.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cl {} failed (exit {})",
            args.join(" "),
            status.code().unwrap_or(-1)
        ))
    }
}

#[cfg(test)]
#[path = "msvc_tests.rs"]
mod tests;
