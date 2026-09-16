//! Typed RecordBatch builder for `python.profile_roofline`.

use std::sync::Arc;

use probing_core::core::{
    ArrayRef, DataType, Field, Float64Array, Int64Array, RecordBatch, Schema, SchemaRef,
    StringArray,
};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::Python;

use super::tbls::{try_record_batch, PythonTableError, TableResult};

fn profile_roofline_schema() -> SchemaRef {
    SchemaRef::new(Schema::new(vec![
        Field::new("capture_id", DataType::Utf8, true),
        Field::new("local_step", DataType::Int64, true),
        Field::new("global_step", DataType::Int64, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("role", DataType::Utf8, true),
        Field::new("op_name", DataType::Utf8, true),
        Field::new("kernel_name", DataType::Utf8, true),
        Field::new("calls", DataType::Int64, true),
        Field::new("self_duration_us", DataType::Int64, true),
        Field::new("flops", DataType::Int64, true),
        Field::new("dram_bytes", DataType::Int64, true),
        Field::new("arithmetic_intensity", DataType::Float64, true),
        Field::new("achieved_flops", DataType::Float64, true),
        Field::new("achieved_bytes_per_sec", DataType::Float64, true),
        Field::new("peak_flops", DataType::Float64, true),
        Field::new("peak_bytes_per_sec", DataType::Float64, true),
        Field::new("peak_flops_kind", DataType::Utf8, true),
        Field::new("boundedness", DataType::Float64, true),
        Field::new("bottleneck", DataType::Utf8, true),
        Field::new("data_quality", DataType::Utf8, true),
    ]))
}

fn dict_opt_i64(dict: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<i64> {
    dict.get_item(key).ok().flatten().and_then(|v| v.extract::<i64>().ok())
}

fn dict_opt_f64(dict: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<f64> {
    dict.get_item(key).ok().flatten().and_then(|v| v.extract::<f64>().ok())
}

fn dict_opt_str(dict: &pyo3::Bound<'_, PyDict>, key: &str) -> Option<String> {
    dict.get_item(key).ok().flatten().and_then(|v| v.extract::<String>().ok())
}

pub fn profile_roofline_batches() -> TableResult<Vec<RecordBatch>> {
    Python::attach(|py| {
        let schema = profile_roofline_schema();
        let module = py.import("probing.profiling.torch_profiler.sql")?;
        let raw = module.call_method0("profile_roofline_rows")?;
        let rows = raw.cast::<PyList>()?;
        if rows.is_empty() {
            let columns: Vec<ArrayRef> = schema
                .fields()
                .iter()
                .map(|field| {
                    match field.data_type() {
                        DataType::Utf8 => Arc::new(StringArray::from(Vec::<Option<String>>::new())) as ArrayRef,
                        DataType::Float64 => Arc::new(Float64Array::from(Vec::<Option<f64>>::new())) as ArrayRef,
                        _ => Arc::new(Int64Array::from(Vec::<Option<i64>>::new())) as ArrayRef,
                    }
                })
                .collect();
            return Ok(vec![try_record_batch(schema, columns)?]);
        }

        let mut capture_id = Vec::new();
        let mut local_step = Vec::new();
        let mut global_step = Vec::new();
        let mut rank = Vec::new();
        let mut role = Vec::new();
        let mut op_name = Vec::new();
        let mut kernel_name = Vec::new();
        let mut calls = Vec::new();
        let mut self_duration_us = Vec::new();
        let mut flops = Vec::new();
        let mut dram_bytes = Vec::new();
        let mut arithmetic_intensity = Vec::new();
        let mut achieved_flops = Vec::new();
        let mut achieved_bytes_per_sec = Vec::new();
        let mut peak_flops = Vec::new();
        let mut peak_bytes_per_sec = Vec::new();
        let mut peak_flops_kind = Vec::new();
        let mut boundedness = Vec::new();
        let mut bottleneck = Vec::new();
        let mut data_quality = Vec::new();

        for item in rows.iter() {
            let dict = item.cast::<PyDict>().map_err(|_| {
                PythonTableError::BatchBuild("profile_roofline row not dict".into())
            })?;
            capture_id.push(dict_opt_str(&dict, "capture_id"));
            local_step.push(dict_opt_i64(&dict, "local_step"));
            global_step.push(dict_opt_i64(&dict, "global_step"));
            rank.push(dict_opt_i64(&dict, "rank"));
            role.push(dict_opt_str(&dict, "role"));
            op_name.push(dict_opt_str(&dict, "op_name"));
            kernel_name.push(dict_opt_str(&dict, "kernel_name"));
            calls.push(dict_opt_i64(&dict, "calls"));
            self_duration_us.push(dict_opt_i64(&dict, "self_duration_us"));
            flops.push(dict_opt_i64(&dict, "flops"));
            dram_bytes.push(dict_opt_i64(&dict, "dram_bytes"));
            arithmetic_intensity.push(dict_opt_f64(&dict, "arithmetic_intensity"));
            achieved_flops.push(dict_opt_f64(&dict, "achieved_flops"));
            achieved_bytes_per_sec.push(dict_opt_f64(&dict, "achieved_bytes_per_sec"));
            peak_flops.push(dict_opt_f64(&dict, "peak_flops"));
            peak_bytes_per_sec.push(dict_opt_f64(&dict, "peak_bytes_per_sec"));
            peak_flops_kind.push(dict_opt_str(&dict, "peak_flops_kind"));
            boundedness.push(dict_opt_f64(&dict, "boundedness"));
            bottleneck.push(dict_opt_str(&dict, "bottleneck"));
            data_quality.push(dict_opt_str(&dict, "data_quality"));
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(capture_id)),
            Arc::new(Int64Array::from(local_step)),
            Arc::new(Int64Array::from(global_step)),
            Arc::new(Int64Array::from(rank)),
            Arc::new(StringArray::from(role)),
            Arc::new(StringArray::from(op_name)),
            Arc::new(StringArray::from(kernel_name)),
            Arc::new(Int64Array::from(calls)),
            Arc::new(Int64Array::from(self_duration_us)),
            Arc::new(Int64Array::from(flops)),
            Arc::new(Int64Array::from(dram_bytes)),
            Arc::new(Float64Array::from(arithmetic_intensity)),
            Arc::new(Float64Array::from(achieved_flops)),
            Arc::new(Float64Array::from(achieved_bytes_per_sec)),
            Arc::new(Float64Array::from(peak_flops)),
            Arc::new(Float64Array::from(peak_bytes_per_sec)),
            Arc::new(StringArray::from(peak_flops_kind)),
            Arc::new(Float64Array::from(boundedness)),
            Arc::new(StringArray::from(bottleneck)),
            Arc::new(StringArray::from(data_quality)),
        ];
        Ok(vec![try_record_batch(schema, columns)?])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_roofline_schema_has_expected_columns() {
        let schema = profile_roofline_schema();
        for col in [
            "capture_id",
            "arithmetic_intensity",
            "boundedness",
            "bottleneck",
            "data_quality",
        ] {
            assert!(schema.field_with_name(col).is_ok(), "missing column {col}");
        }
    }
}
