//! Reference values for the tests, produced once with SciPy by
//! `tests/fixtures/dense-native/make_fixtures.py`.

use serde_json::Value;

pub fn scipy_reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/dense-native/scipy-reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn dims(value: &Value) -> [usize; 3] {
    let entries: Vec<usize> = value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    [entries[0], entries[1], entries[2]]
}

pub fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}
