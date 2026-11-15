//! The chapters of [`gust::guide`](crate::guide), one empty module each (the semver rule: the guide modules hold
//! nothing but chapter modules, which hold nothing; they exist for their documentation). Chapters 3 to 12 are listed in
//! the index as not yet written and have no module until they have text.

#[doc = include_str!("01-why-gust.md")]
pub mod why_gust {}

#[doc = include_str!("02-install-a.md")]
#[doc = include_str!("listings/hello-gust-kernel.rs")]
#[doc = include_str!("02-install-b.md")]
#[doc = include_str!("listings/hello-gust-imports.rs")]
#[doc = ""]
#[doc = include_str!("listings/hello-gust-main.rs")]
#[doc = include_str!("02-install-c.md")]
pub mod install_and_first_program {}
