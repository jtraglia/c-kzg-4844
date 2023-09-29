#![allow(dead_code)]

use crate::bindings::hex_to_bytes;
use crate::kzg_mainnet::Blob;
use crate::{Bytes48, Error};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Input {
    blobs: Vec<String>,
    commitments: Vec<String>,
    proofs: Vec<String>,
}

impl Input {
    pub fn get_blobs(&self) -> Result<Vec<Blob>, Error> {
        let mut v: Vec<Blob> = Vec::new();
        for blob in &self.blobs {
            let blob = Blob::from_bytes(&hex_to_bytes(blob).unwrap())?;
            v.push(blob);
        }
        Ok(v)
    }

    pub fn get_commitments(&self) -> Result<Vec<Bytes48>, Error> {
        self.commitments
            .iter()
            .map(|s| Bytes48::from_hex(s))
            .collect::<Result<Vec<Bytes48>, Error>>()
    }

    pub fn get_proofs(&self) -> Result<Vec<Bytes48>, Error> {
        self.proofs
            .iter()
            .map(|s| Bytes48::from_hex(s))
            .collect::<Result<Vec<Bytes48>, Error>>()
    }
}

#[derive(Deserialize)]
pub struct Test {
    pub input: Input,
    output: Option<bool>,
}

impl Test {
    pub fn get_output(&self) -> Option<bool> {
        self.output
    }
}
