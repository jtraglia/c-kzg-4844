#![allow(dead_code)]

use crate::bindings::hex_to_bytes;
use crate::kzg_mainnet::Blob;
use crate::{Bytes48, Error};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Input<'a> {
    blob: &'a str,
}

impl Input<'_> {
    pub fn get_blob(&self) -> Result<Blob, Error> {
        Blob::from_bytes(&hex_to_bytes(self.blob).unwrap())
    }
}

#[derive(Deserialize)]
pub struct Test<'a> {
    #[serde(borrow)]
    pub input: Input<'a>,
    #[serde(borrow)]
    output: Option<&'a str>,
}

impl Test<'_> {
    pub fn get_output(&self) -> Option<Bytes48> {
        self.output.map(|s| Bytes48::from_hex(s).unwrap())
    }
}
