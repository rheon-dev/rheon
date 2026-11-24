//! A map as a sorted list, so the state's bytes do not depend on the
//! hasher.

use std::collections::HashMap;
use std::hash::BuildHasher;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S: Serializer, H>(m: &HashMap<String, usize, H>, s: S) -> Result<S::Ok, S::Error> {
    let mut v: Vec<(&String, &usize)> = m.iter().collect();
    v.sort();
    v.serialize(s)
}

pub fn deserialize<'de, D: Deserializer<'de>, H: BuildHasher + Default>(
    d: D,
) -> Result<HashMap<String, usize, H>, D::Error> {
    Ok(Vec::<(String, usize)>::deserialize(d)?
        .into_iter()
        .collect())
}
