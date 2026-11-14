//! The Windows data sources of the sampler: PDH, the ToolHelp process table and DXGI. This is the only module of the
//! bench that uses `unsafe`; every block names what it relies on.

use windows::{
    Win32::{
        Foundation::{CloseHandle, FILETIME, HANDLE},
        Graphics::Dxgi::{
            CreateDXGIFactory1, DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE,
            DXGI_ERROR_NOT_FOUND, IDXGIFactory1,
        },
        System::{
            Diagnostics::ToolHelp::{
                CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
                TH32CS_SNAPPROCESS,
            },
            Performance::{
                PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY,
                PDH_MORE_DATA, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
                PdhGetFormattedCounterArrayW, PdhOpenQueryW,
            },
            Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
        },
    },
    core::w,
};

use super::{DxgiAdapter, Luid, Proc, ProcTable, walk_truncation};

/// A PDH query holding the one wildcard counter `\GPU Engine(*)\Utilization Percentage`.
pub struct Pdh {
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
}

fn status(what: &str, code: u32) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{what} failed with PDH status {code:#010x}"))
    }
}

impl Pdh {
    pub fn open() -> Result<Self, String> {
        let mut query = PDH_HQUERY::default();
        let mut counter = PDH_HCOUNTER::default();
        // SAFETY: both out pointers are valid for the call; the query is closed in `Drop` (or right here on failure).
        unsafe {
            status("PdhOpenQueryW", PdhOpenQueryW(None, 0, &mut query))?;
            let added = PdhAddEnglishCounterW(
                query,
                w!("\\GPU Engine(*)\\Utilization Percentage"),
                0,
                &mut counter,
            );
            if let Err(error) = status("PdhAddEnglishCounterW", added) {
                PdhCloseQuery(query);
                return Err(error);
            }
        }
        Ok(Self { query, counter })
    }

    pub fn collect(&self) -> Result<(), String> {
        // SAFETY: `self.query` is the open query of this value.
        status("PdhCollectQueryData", unsafe {
            PdhCollectQueryData(self.query)
        })
    }

    /// Every instance of the counter at the last collection: (instance name, percent, PDH status; 0 and 1 are valid).
    pub fn read(&self) -> Result<Vec<(String, f64, u32)>, String> {
        let mut bytes = 0u32;
        let mut count = 0u32;
        // SAFETY: a null item buffer with a zero size asks PDH for the size it needs (PDH_MORE_DATA).
        let code = unsafe {
            PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, None)
        };
        if code != PDH_MORE_DATA {
            return if code == 0 {
                Ok(Vec::new())
            } else {
                Err(format!(
                    "PdhGetFormattedCounterArrayW (size) failed with PDH status {code:#010x}"
                ))
            };
        }
        // The buffer holds the item array followed by the names it points into; 8-byte alignment suits the items.
        // PDH may need a little more on the second call when instances appear in between, so retry a few times.
        for _ in 0..4 {
            let mut buffer = vec![0u64; (bytes as usize).div_ceil(8) + 16];
            let mut size = (buffer.len() * 8) as u32;
            let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
            // SAFETY: `items` points at `size` writable bytes, 8-byte aligned, and outlives the reads below.
            let code = unsafe {
                PdhGetFormattedCounterArrayW(
                    self.counter,
                    PDH_FMT_DOUBLE,
                    &mut size,
                    &mut count,
                    Some(items),
                )
            };
            if code == PDH_MORE_DATA {
                bytes = size;
                continue;
            }
            status("PdhGetFormattedCounterArrayW", code)?;
            let mut out = Vec::with_capacity(count as usize);
            for index in 0..count as usize {
                // SAFETY: PDH filled `count` items at the start of the buffer; each name pointer addresses a
                // NUL-terminated wide string inside that same buffer.
                let (name, value, item_status) = unsafe {
                    let item = &*items.add(index);
                    let name = item.szName.to_string().unwrap_or_default();
                    (
                        name,
                        item.FmtValue.Anonymous.doubleValue,
                        item.FmtValue.CStatus,
                    )
                };
                out.push((name, value, item_status));
            }
            return Ok(out);
        }
        Err("PdhGetFormattedCounterArrayW kept asking for more data".to_owned())
    }
}

impl Drop for Pdh {
    fn drop(&mut self) {
        // SAFETY: the query is closed exactly once, here.
        unsafe {
            PdhCloseQuery(self.query);
        }
    }
}

fn filetime(time: FILETIME) -> u64 {
    (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
}

/// (CPU time in 100 ns, creation time) of a process, or `None` when it cannot be opened for query.
fn times(pid: u32) -> Option<(u64, u64)> {
    // SAFETY: the handle is used only for `GetProcessTimes` and closed before returning.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let (mut created, mut exited, mut kernel, mut user) = (
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
        );
        let ok = GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user).is_ok();
        let _ = CloseHandle(handle);
        ok.then(|| (filetime(kernel) + filetime(user), filetime(created)))
    }
}

/// The process table: pid, parent pid, image name, CPU time and creation time (the fields of `Win32_Process`).
pub fn process_table() -> Result<ProcTable, String> {
    // SAFETY: `entry.dwSize` is set as the API requires; the snapshot handle is closed before returning.
    unsafe {
        let snapshot: HANDLE = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            .map_err(|error| format!("CreateToolhelp32Snapshot: {error}"))?;
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut table = Vec::new();
        let mut next = Process32FirstW(snapshot, &mut entry);
        while next.is_ok() {
            let end = entry
                .szExeFile
                .iter()
                .position(|c| *c == 0)
                .unwrap_or(entry.szExeFile.len());
            let pid = entry.th32ProcessID;
            let measured = times(pid);
            table.push(Proc {
                pid,
                ppid: entry.th32ParentProcessID,
                name: String::from_utf16_lossy(&entry.szExeFile[..end]),
                cpu_100ns: measured.map(|(cpu, _)| cpu),
                created: measured.map(|(_, created)| created),
            });
            next = Process32NextW(snapshot, &mut entry);
        }
        // Only ERROR_NO_MORE_FILES is the end of the table; anything else leaves it truncated.
        let truncated = next.err().and_then(|error| walk_truncation(error.code().0));
        let _ = CloseHandle(snapshot);
        Ok(ProcTable {
            rows: table,
            truncated,
        })
    }
}

/// Every DXGI adapter, with its LUID.
pub fn dxgi_adapters() -> Result<Vec<DxgiAdapter>, String> {
    // SAFETY: plain COM calls on interfaces the `windows` crate keeps alive.
    unsafe {
        let factory: IDXGIFactory1 =
            CreateDXGIFactory1().map_err(|error| format!("CreateDXGIFactory1: {error}"))?;
        let mut adapters = Vec::new();
        let mut index = 0;
        loop {
            let adapter = match factory.EnumAdapters1(index) {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(format!("EnumAdapters1({index}): {error}")),
            };
            let desc: DXGI_ADAPTER_DESC1 = adapter
                .GetDesc1()
                .map_err(|error| format!("GetDesc1: {error}"))?;
            let end = desc
                .Description
                .iter()
                .position(|c| *c == 0)
                .unwrap_or(desc.Description.len());
            adapters.push(DxgiAdapter {
                index,
                description: String::from_utf16_lossy(&desc.Description[..end]),
                vendor_id: desc.VendorId,
                device_id: desc.DeviceId,
                luid: Luid {
                    high: desc.AdapterLuid.HighPart as u32,
                    low: desc.AdapterLuid.LowPart,
                },
                software: desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
            });
            index += 1;
        }
        Ok(adapters)
    }
}
